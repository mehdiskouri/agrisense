# ---------------------------------------------------------------------------
# Yield regression forecaster — FAO stress-coefficient + ridge residual
# (GPU-first)
# ---------------------------------------------------------------------------

"""
    compute_derived_features(graph) -> AbstractMatrix{Float32}

Compute yield-specific derived feature columns (on device):
  1. Cumulative DLI — sum of `:lighting` history col 2 over valid ring-buffer entries
  2. Soil health score — normalised composite of moisture, pH, conductivity, temp

Returns an nv × k matrix (k = number of available derived features, 0-2).
"""
function compute_derived_features(graph::LayeredHyperGraph)::AbstractMatrix{Float32}
    nv = graph.n_vertices
    first_layer = first(values(graph.layers))
    backend = array_backend(first_layer.vertex_features)
    cols = AbstractMatrix{Float32}[]

    # --- Cumulative DLI (sum of lighting history column 2) ---
    if haskey(graph.layers, :lighting)
        ll = graph.layers[:lighting]
        buf_d = size(ll.feature_history, 2)
        if buf_d >= 2 && ll.history_length > 0
            hlength = min(ll.history_length, size(ll.feature_history, 3))
            # Sum across valid ring-buffer slots for DLI column (col 2)
            # feature_history is (nv, d, buf_size); slice col 2, valid slots
            dli_hist = ll.feature_history[:, 2, 1:hlength]   # (nv, hlength)
            cum_dli = sum(dli_hist; dims=2)                   # (nv, 1)
            push!(cols, Float32.(cum_dli))
        end
    end

    # --- Soil health score (composite 0-1) ---
    if haskey(graph.layers, :soil)
        sl = graph.layers[:soil]
        d_soil = size(sl.vertex_features, 2)
        if d_soil >= 4
            moisture     = sl.vertex_features[:, 1]
            temperature  = sl.vertex_features[:, 2]
            conductivity = sl.vertex_features[:, 3]
            ph           = sl.vertex_features[:, 4]

            # Each sub-score normalised via sigmoid-like clamp into [0, 1]:
            #   moisture:     optimal near 0.3; score = 1 - |m - 0.3| / 0.3
            #   temperature:  optimal 20-25 °C; piecewise ramp
            #   pH:           optimal 6.0-7.0;  1 - |pH - 6.5| / 3.5
            #   conductivity: lower is better; 1 - clamp(c / 4, 0, 1)
            m_score = @. clamp(1.0f0 - abs(moisture - 0.3f0) / 0.3f0, 0.0f0, 1.0f0)
            t_score = @. clamp(1.0f0 - abs(temperature - 22.5f0) / 17.5f0, 0.0f0, 1.0f0)
            p_score = @. clamp(1.0f0 - abs(ph - 6.5f0) / 3.5f0, 0.0f0, 1.0f0)
            c_score = @. clamp(1.0f0 - conductivity / 4.0f0, 0.0f0, 1.0f0)

            health = @. 0.3f0 * m_score + 0.25f0 * t_score + 0.25f0 * p_score + 0.2f0 * c_score
            push!(cols, reshape(health, nv, 1))
        end
    end

    if isempty(cols)
        return backend isa CPU ? zeros(Float32, nv, 0) :
               (HAS_CUDA ? CUDA.zeros(Float32, nv, 0) : zeros(Float32, nv, 0))
    end
    return hcat(cols...)
end

@kernel function fao_yield_kernel!(y_pred, y_potential, ks, kn, kl, kw)
    i = @index(Global)
    @inbounds begin
        y_pred[i] = y_potential[i] * ks[i] * kn[i] * kl[i] * kw[i]
    end
end

@kernel function nutrient_stress_kernel!(kn, npk_n, npk_p, npk_k,
                                          req_n, req_p, req_k)
    i = @index(Global)
    @inbounds begin
        count = 0.0f0
        total_dr = 0.0f0
        rn = req_n[i]; rp = req_p[i]; rk = req_k[i]
        if rn > 0.0f0
            total_dr += max(rn - npk_n[i], 0.0f0) / rn
            count += 1.0f0
        end
        if rp > 0.0f0
            total_dr += max(rp - npk_p[i], 0.0f0) / rp
            count += 1.0f0
        end
        if rk > 0.0f0
            total_dr += max(rk - npk_k[i], 0.0f0) / rk
            count += 1.0f0
        end
        if count > 0.0f0
            kn[i] = clamp(1.0f0 - total_dr / count, 0.0f0, 1.0f0)
        else
            kn[i] = 1.0f0
        end
    end
end

@kernel function weather_stress_kernel!(kw, temp)
    i = @index(Global)
    @inbounds begin
        t = temp[i]
        if t < 5.0f0
            kw[i] = 0.0f0
        elseif t < 15.0f0
            kw[i] = (t - 5.0f0) / 10.0f0
        elseif t <= 30.0f0
            kw[i] = 1.0f0
        elseif t < 40.0f0
            kw[i] = (40.0f0 - t) / 10.0f0
        else
            kw[i] = 0.0f0
        end
    end
end

# ---------------------------------------------------------------------------
# Persistent residual model coefficients (module-level state)
# ---------------------------------------------------------------------------
const RESIDUAL_COEFFICIENTS = Ref{Union{Nothing, Vector{Float32}}}(nothing)

# ---------------------------------------------------------------------------
# Stress coefficient computations (GPU-accelerated)
# ---------------------------------------------------------------------------

"""
    compute_stress_coefficients(graph) -> (Ks, Kn, Kl, Kw)

Returns four vectors of length `n_vertices`, each ∈ [0, 1].
Stays on whatever device the graph data lives on.
"""
function compute_stress_coefficients(graph::LayeredHyperGraph)
    nv = graph.n_vertices
    # Detect backend from first available layer
    first_layer = first(values(graph.layers))
    backend = array_backend(first_layer.vertex_features)
    _ones = backend isa CPU ? ones(Float32, nv) :
            (HAS_CUDA ? CUDA.ones(Float32, nv) : ones(Float32, nv))

    # --- Water stress Ks (GPU broadcast) ---
    ks = copy(_ones)
    if haskey(graph.layers, :soil)
        moisture = graph.layers[:soil].vertex_features[:, 1]
        ks = @. clamp((moisture - DEFAULT_WILTING_POINT) /
                       (DEFAULT_FIELD_CAPACITY - DEFAULT_WILTING_POINT), 0.0f0, 1.0f0)
    end

    # --- Nutrient stress Kn (GPU kernel) ---
    kn = copy(_ones)
    if haskey(graph.layers, :npk) && haskey(graph.layers, :crop_requirements)
        npk_feat  = graph.layers[:npk].vertex_features
        crop_feat = graph.layers[:crop_requirements].vertex_features
        launch_kernel!(nutrient_stress_kernel!, backend, nv,
                       kn,
                       npk_feat[:, 1], npk_feat[:, 2], npk_feat[:, 3],
                       crop_feat[:, 3], crop_feat[:, 4], crop_feat[:, 5])
    end

    # --- Light stress Kl (GPU broadcast) ---
    kl = copy(_ones)
    if haskey(graph.layers, :lighting)
        dli = graph.layers[:lighting].vertex_features[:, 2]
        optimal_dli = 20.0f0
        kl = @. clamp(dli / optimal_dli, 0.0f0, 1.0f0)
    end

    # --- Weather stress Kw (GPU kernel — piecewise linear) ---
    kw = copy(_ones)
    if haskey(graph.layers, :weather)
        temp = graph.layers[:weather].vertex_features[:, 1]
        launch_kernel!(weather_stress_kernel!, backend, nv, kw, temp)
    end

    return ks, kn, kl, kw
end

# ---------------------------------------------------------------------------
# Ridge regression for residual correction (always CPU — small matrices)
# ---------------------------------------------------------------------------

"""
    fit_residual_model(X, y_residual; λ=1.0f0) -> Vector{Float32}

Ridge regression: β = (X'X + λI) \\ (X'y).  Always runs on CPU.
"""
function fit_residual_model(X::AbstractMatrix{Float32}, y_residual::AbstractVector{Float32};
                            λ::Float32=1.0f0)::Vector{Float32}
    Xc = ensure_cpu(X)
    yc = ensure_cpu(y_residual)
    XtX = Xc' * Xc
    Xty = Xc' * yc
    β = (XtX + λ * I) \ Xty
    return Float32.(β)
end

"""
    train_yield_residual!(graph, actual_yields) -> nothing

Fit the residual model using actual yield data and store coefficients.
`actual_yields` is a `Dict{String,Float32}` mapping vertex_id → observed yield.
"""
function train_yield_residual!(graph::LayeredHyperGraph,
                                actual_yields::Dict{String,Float32})
    nv = graph.n_vertices

    # Assemble feature matrix (may be on GPU) + derived features
    feature_layers = Symbol[]
    for lsym in [:soil, :lighting, :crop_requirements, :vision]
        haskey(graph.layers, lsym) && push!(feature_layers, lsym)
    end
    X_raw = multi_layer_features(graph, feature_layers)
    X_derived = compute_derived_features(graph)
    X = size(X_derived, 2) > 0 ? hcat(X_raw, X_derived) : X_raw

    # FAO predictions (on device)
    ks, kn, kl, kw = compute_stress_coefficients(graph)
    y_pot = if haskey(graph.layers, :crop_requirements)
        graph.layers[:crop_requirements].vertex_features[:, 1]
    else
        first_layer = first(values(graph.layers))
        backend = array_backend(first_layer.vertex_features)
        backend isa CPU ? ones(Float32, nv) :
            (HAS_CUDA ? CUDA.ones(Float32, nv) : ones(Float32, nv))
    end
    y_fao = @. y_pot * ks * kn * kl * kw

    # Pull to CPU for residual building
    y_fao_cpu = ensure_cpu(y_fao)
    X_cpu = ensure_cpu(X)

    obs_indices = Int[]
    y_res = Float32[]
    for (vid, actual) in actual_yields
        haskey(graph.vertex_index, vid) || continue
        idx = graph.vertex_index[vid]
        push!(obs_indices, idx)
        push!(y_res, actual - y_fao_cpu[idx])
    end

    if length(obs_indices) < size(X_cpu, 2) + 1
        @warn "Not enough observations ($(length(obs_indices))) for ridge regression — " *
              "need at least $(size(X_cpu, 2) + 1). Coefficients not updated."
        return nothing
    end

    X_obs = X_cpu[obs_indices, :]
    β = fit_residual_model(X_obs, Float32.(y_res))
    RESIDUAL_COEFFICIENTS[] = β
    return nothing
end

"""
    compute_yield_forecast(graph::LayeredHyperGraph) -> Vector{Dict}

Per-bed yield estimate via FAO stress model + optional ridge residual correction.
GPU-accelerated: stress coefficients computed on device, pulled to CPU for Dict output.
"""
function compute_yield_forecast(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    has_crop = haskey(graph.layers, :crop_requirements)
    !has_crop && return Dict{String,Any}[]

    nv = graph.n_vertices
    crop_layer = graph.layers[:crop_requirements]

    # Potential yield from crop requirements col 1 (on device)
    y_pot = crop_layer.vertex_features[:, 1]

    # Stress coefficients (on device)
    ks, kn, kl, kw = compute_stress_coefficients(graph)

    # FAO base prediction (GPU broadcast)
    y_fao = @. y_pot * ks * kn * kl * kw

    # Ridge residual correction
    y_final = copy(y_fao)
    model_layer = "fao_only"
    if RESIDUAL_COEFFICIENTS[] !== nothing
        feature_layers = Symbol[]
        for lsym in [:soil, :lighting, :crop_requirements, :vision]
            haskey(graph.layers, lsym) && push!(feature_layers, lsym)
        end
        X_raw = multi_layer_features(graph, feature_layers)
        X_derived = compute_derived_features(graph)
        X = size(X_derived, 2) > 0 ? hcat(X_raw, X_derived) : X_raw
        β = RESIDUAL_COEFFICIENTS[]
        # Pull to CPU for matmul with β (small vector, always CPU)
        X_cpu = ensure_cpu(X)
        if size(X_cpu, 2) == length(β)
            y_residual_cpu = X_cpu * β
            y_fao_cpu = ensure_cpu(y_fao)
            y_final_cpu = y_fao_cpu .+ y_residual_cpu
            # We'll use CPU from here since we need Dict building anyway
            y_final = y_final_cpu
            model_layer = "fao_plus_residual"
        end
    end

    # Confidence intervals: ±20% for FAO-only, ±10% with residual
    ci_factor = model_layer == "fao_only" ? 0.20f0 : 0.10f0

    # Pull everything to CPU for Dict building
    y_final_cpu = ensure_cpu(y_final)
    ks_cpu = ensure_cpu(ks)
    kn_cpu = ensure_cpu(kn)
    kl_cpu = ensure_cpu(kl)
    kw_cpu = ensure_cpu(kw)

    # Aggregate per crop bed via :crop_requirements edges
    results = Dict{String,Any}[]
    B_cpu = ensure_cpu(crop_layer.incidence)
    ne = size(B_cpu, 2)
    for e in 1:ne
        members = findall(!iszero, @view B_cpu[:, e])
        isempty(members) && continue

        bed_id = e <= length(crop_layer.edge_ids) ?
                 crop_layer.edge_ids[e] : "bed_$e"

        est = mean(y_final_cpu[members])
        push!(results, Dict{String,Any}(
            "crop_bed_id" => bed_id,
            "yield_estimate_kg_m2" => Float64(est),
            "yield_lower" => Float64(est * (1.0f0 - ci_factor)),
            "yield_upper" => Float64(est * (1.0f0 + ci_factor)),
            "confidence" => Float64(1.0f0 - ci_factor),
            "stress_factors" => Dict{String,Any}(
                "Ks" => Float64(mean(ks_cpu[members])),
                "Kn" => Float64(mean(kn_cpu[members])),
                "Kl" => Float64(mean(kl_cpu[members])),
                "Kw" => Float64(mean(kw_cpu[members])),
            ),
            "model_layer" => model_layer,
        ))
    end

    return results
end
