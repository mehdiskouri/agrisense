# ---------------------------------------------------------------------------
# Yield regression forecaster — FAO stress-coefficient + ridge residual
# ---------------------------------------------------------------------------

@kernel function fao_yield_kernel!(y_pred, y_potential, ks, kn, kl, kw)
    i = @index(Global)
    @inbounds begin
        y_pred[i] = y_potential[i] * ks[i] * kn[i] * kl[i] * kw[i]
    end
end

# ---------------------------------------------------------------------------
# Persistent residual model coefficients (module-level state)
# ---------------------------------------------------------------------------
const RESIDUAL_COEFFICIENTS = Ref{Union{Nothing, Vector{Float32}}}(nothing)

# ---------------------------------------------------------------------------
# Stress coefficient computations
# ---------------------------------------------------------------------------

"""
    compute_stress_coefficients(graph) -> (Ks, Kn, Kl, Kw)

Returns four Float32 vectors of length `n_vertices`, each ∈ [0, 1].
"""
function compute_stress_coefficients(graph::LayeredHyperGraph)
    nv = graph.n_vertices

    # --- Water stress Ks ---
    ks = ones(Float32, nv)
    if haskey(graph.layers, :soil)
        moisture = Float32.(graph.layers[:soil].vertex_features[:, 1])
        # Linear ramp: 1.0 at field capacity (0.35), 0.0 at wilting point (0.15)
        @. ks = clamp((moisture - DEFAULT_WILTING_POINT) /
                       (DEFAULT_FIELD_CAPACITY - DEFAULT_WILTING_POINT), 0.0f0, 1.0f0)
    end

    # --- Nutrient stress Kn ---
    kn = ones(Float32, nv)
    if haskey(graph.layers, :npk) && haskey(graph.layers, :crop_requirements)
        npk_feat = graph.layers[:npk].vertex_features
        crop_feat = graph.layers[:crop_requirements].vertex_features
        for v in 1:nv
            # Mean deficit ratio across N, P, K
            deficits = Float32[]
            for (cur_col, req_col) in [(1, 3), (2, 4), (3, 5)]
                req = crop_feat[v, req_col]
                if req > 0
                    deficit_ratio = max(req - npk_feat[v, cur_col], 0.0f0) / req
                    push!(deficits, deficit_ratio)
                end
            end
            if !isempty(deficits)
                kn[v] = clamp(1.0f0 - mean(deficits), 0.0f0, 1.0f0)
            end
        end
    end

    # --- Light stress Kl ---
    kl = ones(Float32, nv)
    if haskey(graph.layers, :lighting)
        dli = Float32.(graph.layers[:lighting].vertex_features[:, 2])  # col 2 = DLI
        # DLI requirement ~ 20 mol/m²/d for most crops; ratio capped at 1.0
        optimal_dli = 20.0f0
        @. kl = clamp(dli / optimal_dli, 0.0f0, 1.0f0)
    end

    # --- Weather stress Kw ---
    kw = ones(Float32, nv)
    if haskey(graph.layers, :weather)
        temp = Float32.(graph.layers[:weather].vertex_features[:, 1])
        # Optimal range 15-30°C, linear decay outside
        for v in 1:nv
            t = temp[v]
            if t < 5.0f0
                kw[v] = 0.0f0
            elseif t < 15.0f0
                kw[v] = (t - 5.0f0) / 10.0f0
            elseif t <= 30.0f0
                kw[v] = 1.0f0
            elseif t < 40.0f0
                kw[v] = (40.0f0 - t) / 10.0f0
            else
                kw[v] = 0.0f0
            end
        end
    end

    return ks, kn, kl, kw
end

# ---------------------------------------------------------------------------
# Ridge regression for residual correction
# ---------------------------------------------------------------------------

"""
    fit_residual_model(X, y_residual; λ=1.0f0) -> Vector{Float32}

Ridge regression: β = (X'X + λI) \\ (X'y)
"""
function fit_residual_model(X::Matrix{Float32}, y_residual::Vector{Float32};
                            λ::Float32=1.0f0)::Vector{Float32}
    p = size(X, 2)
    XtX = X' * X
    Xty = X' * y_residual
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

    # Assemble feature matrix
    feature_layers = Symbol[]
    for lsym in [:soil, :lighting, :crop_requirements, :vision]
        haskey(graph.layers, lsym) && push!(feature_layers, lsym)
    end
    X = multi_layer_features(graph, feature_layers)

    # FAO predictions
    ks, kn, kl, kw = compute_stress_coefficients(graph)
    y_pot = if haskey(graph.layers, :crop_requirements)
        Float32.(graph.layers[:crop_requirements].vertex_features[:, 1])
    else
        ones(Float32, nv)
    end
    y_fao = y_pot .* ks .* kn .* kl .* kw

    # Build target residual vector for vertices with observations
    # Use only vertices with known actuals
    obs_indices = Int[]
    y_res = Float32[]
    for (vid, actual) in actual_yields
        haskey(graph.vertex_index, vid) || continue
        idx = graph.vertex_index[vid]
        push!(obs_indices, idx)
        push!(y_res, actual - y_fao[idx])
    end

    if length(obs_indices) < size(X, 2) + 1
        @warn "Not enough observations ($(length(obs_indices))) for ridge regression — " *
              "need at least $(size(X, 2) + 1). Coefficients not updated."
        return nothing
    end

    X_obs = X[obs_indices, :]
    β = fit_residual_model(X_obs, Float32.(y_res))
    RESIDUAL_COEFFICIENTS[] = β
    return nothing
end

"""
    compute_yield_forecast(graph::LayeredHyperGraph) -> Vector{Dict}

Per-bed yield estimate via FAO stress model + optional ridge residual correction.
"""
function compute_yield_forecast(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    has_crop = haskey(graph.layers, :crop_requirements)
    !has_crop && return Dict{String,Any}[]

    nv = graph.n_vertices
    crop_layer = graph.layers[:crop_requirements]

    # Potential yield from crop requirements col 1
    y_pot = Float32.(crop_layer.vertex_features[:, 1])

    # Stress coefficients
    ks, kn, kl, kw = compute_stress_coefficients(graph)

    # FAO base prediction (CPU — graph data is always on CPU)
    y_fao = @. y_pot * ks * kn * kl * kw

    # Ridge residual correction
    y_final = copy(y_fao)
    model_layer = "fao_only"
    if RESIDUAL_COEFFICIENTS[] !== nothing
        feature_layers = Symbol[]
        for lsym in [:soil, :lighting, :crop_requirements, :vision]
            haskey(graph.layers, lsym) && push!(feature_layers, lsym)
        end
        X = multi_layer_features(graph, feature_layers)
        β = RESIDUAL_COEFFICIENTS[]
        if size(X, 2) == length(β)
            y_residual = X * β
            y_final .= y_fao .+ y_residual
            model_layer = "fao_plus_residual"
        end
    end

    # Confidence intervals: ±20% for FAO-only, ±10% with residual
    ci_factor = model_layer == "fao_only" ? 0.20f0 : 0.10f0

    # Aggregate per crop bed via :crop_requirements edges
    results = Dict{String,Any}[]
    ne = size(crop_layer.incidence, 2)
    for e in 1:ne
        members = findall(!iszero, @view crop_layer.incidence[:, e])
        isempty(members) && continue

        bed_id = e <= length(crop_layer.edge_ids) ?
                 crop_layer.edge_ids[e] : "bed_$e"

        est = mean(y_final[members])
        push!(results, Dict{String,Any}(
            "crop_bed_id" => bed_id,
            "yield_estimate_kg_m2" => Float64(est),
            "yield_lower" => Float64(est * (1.0f0 - ci_factor)),
            "yield_upper" => Float64(est * (1.0f0 + ci_factor)),
            "confidence" => Float64(1.0f0 - ci_factor),
            "stress_factors" => Dict{String,Any}(
                "Ks" => Float64(mean(ks[members])),
                "Kn" => Float64(mean(kn[members])),
                "Kl" => Float64(mean(kl[members])),
                "Kw" => Float64(mean(kw[members])),
            ),
            "model_layer" => model_layer,
        ))
    end

    return results
end
