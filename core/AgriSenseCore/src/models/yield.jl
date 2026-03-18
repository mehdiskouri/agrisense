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
            # NaN guard: replace NaN with 0 before summing (Phase 13)
            dli_safe = @. ifelse(isnan(dli_hist), 0.0f0, dli_hist)
            cum_dli = sum(dli_safe; dims=2)                   # (nv, 1)
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
            # NaN guard: replace NaN vertex_features with neutral scores (Phase 13)
            safe_m = @. ifelse(isnan(moisture), 0.3f0, moisture)
            safe_t = @. ifelse(isnan(temperature), 22.5f0, temperature)
            safe_p = @. ifelse(isnan(ph), 6.5f0, ph)
            safe_c = @. ifelse(isnan(conductivity), 0.0f0, conductivity)
            m_score = @. clamp(1.0f0 - abs(safe_m - 0.3f0) / 0.3f0, 0.0f0, 1.0f0)
            t_score = @. clamp(1.0f0 - abs(safe_t - 22.5f0) / 17.5f0, 0.0f0, 1.0f0)
            p_score = @. clamp(1.0f0 - abs(safe_p - 6.5f0) / 3.5f0, 0.0f0, 1.0f0)
            c_score = @. clamp(1.0f0 - safe_c / 4.0f0, 0.0f0, 1.0f0)

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
        # NaN guard: treat NaN nutrient reading as 0 (worst case)
        nn = npk_n[i]; nn = isnan(nn) ? 0.0f0 : nn
        np_ = npk_p[i]; np_ = isnan(np_) ? 0.0f0 : np_
        nk = npk_k[i]; nk = isnan(nk) ? 0.0f0 : nk
        if rn > 0.0f0
            total_dr += max(rn - nn, 0.0f0) / rn
            count += 1.0f0
        end
        if rp > 0.0f0
            total_dr += max(rp - np_, 0.0f0) / rp
            count += 1.0f0
        end
        if rk > 0.0f0
            total_dr += max(rk - nk, 0.0f0) / rk
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
        # NaN guard: unknown temp → no stress assumed
        if isnan(t)
            kw[i] = 1.0f0
        elseif t < 5.0f0
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
# Per-farm residual model coefficients (keyed by farm_id)
# ---------------------------------------------------------------------------
const RESIDUAL_COEFF_CACHE = Dict{String, Vector{Float32}}()
const RESIDUAL_STD_CACHE = Dict{String, Float32}()

"""
    get_residual_coefficients(farm_id) -> Union{Nothing, Vector{Float32}}

Return stored ridge coefficients for `farm_id`, or `nothing` if not trained.
"""
function get_residual_coefficients(farm_id::String)::Union{Nothing, Vector{Float32}}
    return get(RESIDUAL_COEFF_CACHE, farm_id, nothing)
end

"""
    get_residual_std(farm_id) -> Union{Nothing, Float32}

Return stored residual standard deviation for `farm_id`, or `nothing`.
"""
function get_residual_std(farm_id::String)::Union{Nothing, Float32}
    return get(RESIDUAL_STD_CACHE, farm_id, nothing)
end

"""
    evict_residual!(farm_id) -> Bool

Remove residual model state for `farm_id`. Returns `true` if it was present.
"""
function evict_residual!(farm_id::String)::Bool
    had = haskey(RESIDUAL_COEFF_CACHE, farm_id)
    delete!(RESIDUAL_COEFF_CACHE, farm_id)
    delete!(RESIDUAL_STD_CACHE, farm_id)
    return had
end

"""
    clear_residual_cache!() -> Nothing

Remove all residual model state.
"""
function clear_residual_cache!()
    empty!(RESIDUAL_COEFF_CACHE)
    empty!(RESIDUAL_STD_CACHE)
    return nothing
end

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

    # --- Water stress Ks (GPU broadcast) — NaN → Ks=1.0 (no stress assumed) ---
    ks = copy(_ones)
    if haskey(graph.layers, :soil)
        moisture = graph.layers[:soil].vertex_features[:, 1]
        safe_m = @. ifelse(isnan(moisture), 0.25f0, moisture)
        ks = @. clamp((safe_m - DEFAULT_WILTING_POINT) /
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

    # --- Light stress Kl (GPU broadcast) — NaN DLI → Kl=1.0 (no stress assumed) ---
    kl = copy(_ones)
    if haskey(graph.layers, :lighting)
        dli = graph.layers[:lighting].vertex_features[:, 2]
        optimal_dli = 20.0f0
        safe_dli = @. ifelse(isnan(dli), optimal_dli, dli)
        kl = @. clamp(safe_dli / optimal_dli, 0.0f0, 1.0f0)
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
# Ridge regression for residual correction (GPU when available)
# ---------------------------------------------------------------------------

"""
    fit_residual_model(X, y_residual; λ=1.0f0) -> Vector{Float32}

Ridge regression: β = (X'X + λI) \\ (X'y).
Uses GPU solve when `X` is CUDA-backed and CPU solve otherwise.
"""
function fit_residual_model(X::AbstractMatrix{Float32}, y_residual::AbstractVector{Float32};
                            λ::Float32=1.0f0)::Vector{Float32}
    # ── NaN guard: filter out rows containing any NaN ──────────────
    Xc_raw = ensure_cpu(X)
    yc_raw = ensure_cpu(y_residual)
    n_rows = size(Xc_raw, 1)
    valid = trues(n_rows)
    for r in 1:n_rows
        if isnan(yc_raw[r])
            valid[r] = false; continue
        end
        for c in axes(Xc_raw, 2)
            if isnan(Xc_raw[r, c])
                valid[r] = false; break
            end
        end
    end
    Xc_clean = Xc_raw[valid, :]
    yc_clean = yc_raw[valid]

    n_features = size(Xc_clean, 2)
    # If nothing valid, return zero coefficients
    if size(Xc_clean, 1) == 0
        return zeros(Float32, n_features)
    end

    backend = array_backend(X)
    if !(backend isa CPU) && HAS_CUDA
        Xg = CuArray(Xc_clean)
        yg = CuArray(yc_clean)
        XtX = Xg' * Xg
        Xty = Xg' * yg
        I_gpu = CuArray(Matrix{Float32}(I, n_features, n_features))
        β_gpu = (XtX + λ * I_gpu) \ Xty
        return Float32.(ensure_cpu(β_gpu))
    end

    XtX = Xc_clean' * Xc_clean
    Xty = Xc_clean' * yc_clean
    β = (XtX + λ * I) \ Xty
    return Float32.(β)
end

"""
    train_yield_residual!(graph, actual_yields) -> Bool

Fit the residual model using actual yield data and store coefficients.
`actual_yields` is a `Dict{String,Float32}` mapping vertex_id → observed yield.
Returns `true` when coefficients are fitted, `false` when data is insufficient.
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

    # Pull FAO baseline to CPU for observed residual target construction
    y_fao_cpu = ensure_cpu(y_fao)
    n_features = size(X, 2)

    obs_indices = Int[]
    y_res = Float32[]
    for (vid, actual) in actual_yields
        haskey(graph.vertex_index, vid) || continue
        idx = graph.vertex_index[vid]
        push!(obs_indices, idx)
        push!(y_res, actual - y_fao_cpu[idx])
    end

    farm_id = graph.farm_id

    if length(obs_indices) < n_features + 1
        # Insufficient labels is a normal condition for new farms; keep residual model disabled.
        delete!(RESIDUAL_COEFF_CACHE, farm_id)
        delete!(RESIDUAL_STD_CACHE, farm_id)
        return false
    end

    X_obs = X[obs_indices, :]
    y_res_vec = Float32.(y_res)
    β = fit_residual_model(X_obs, y_res_vec)

    # Residual std used for confidence interval construction
    X_obs_cpu = ensure_cpu(X_obs)
    y_hat = X_obs_cpu * β
    resid = y_res_vec .- y_hat
    sigma = Float32(std(resid; corrected=true))

    RESIDUAL_COEFF_CACHE[farm_id] = β
    RESIDUAL_STD_CACHE[farm_id] = isfinite(sigma) ? max(sigma, 1.0f-4) : 1.0f-4
    return true
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

    # Ridge residual correction (per-farm coefficients)
    farm_id = graph.farm_id
    y_final = copy(y_fao)
    model_layer = "fao_only"
    β_farm = get_residual_coefficients(farm_id)
    if β_farm !== nothing
        feature_layers = Symbol[]
        for lsym in [:soil, :lighting, :crop_requirements, :vision]
            haskey(graph.layers, lsym) && push!(feature_layers, lsym)
        end
        X_raw = multi_layer_features(graph, feature_layers)
        X_derived = compute_derived_features(graph)
        X = size(X_derived, 2) > 0 ? hcat(X_raw, X_derived) : X_raw
        # Pull to CPU for matmul with β (small vector, always CPU)
        X_cpu = ensure_cpu(X)
        if size(X_cpu, 2) == length(β_farm)
            y_residual_cpu = X_cpu * β_farm
            y_fao_cpu = ensure_cpu(y_fao)
            y_final_cpu = y_fao_cpu .+ y_residual_cpu
            # We'll use CPU from here since we need Dict building anyway
            y_final = y_final_cpu
            model_layer = "fao_plus_residual"
        end
    end

    # Confidence intervals:
    # - residual model: Gaussian 95% CI using trained residual std
    # - fao_only: conservative proportional fallback
    σ_farm = get_residual_std(farm_id)
    use_residual_ci = model_layer == "fao_plus_residual" && σ_farm !== nothing
    z95 = 1.96f0
    residual_sigma = use_residual_ci ? Float32(σ_farm) : 0.0f0

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
        half_width = use_residual_ci ? z95 * residual_sigma : 0.20f0 * abs(Float32(est))
        lower = max(0.0f0, Float32(est) - half_width)
        upper = Float32(est) + half_width
        conf = use_residual_ci ? 0.95 : 0.80
        push!(results, Dict{String,Any}(
            "crop_bed_id" => bed_id,
            "yield_estimate_kg_m2" => Float64(est),
            "yield_lower" => Float64(lower),
            "yield_upper" => Float64(upper),
            "confidence" => Float64(conf),
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

"""
    compute_yield_forecast_single(graph) -> Vector{Dict}

Explicit alias for the existing single-model forecast path.
"""
function compute_yield_forecast_single(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    return compute_yield_forecast(graph)
end

# ---------------------------------------------------------------------------
# Ensemble yield forecasting helpers
# ---------------------------------------------------------------------------

"""Per-farm ensemble blending weights keyed by `farm_id`."""
const ENSEMBLE_WEIGHT_CACHE = Dict{String, Vector{Float32}}()

"""Per-farm tuned hyperparameters keyed by `farm_id`."""
const ENSEMBLE_HYPERPARAM_CACHE = Dict{String, Dict{String,Float32}}()

function _normalise_weights(weights::Vector{Float32})::Vector{Float32}
    isempty(weights) && return Float32[1 / 3, 1 / 3, 1 / 3]
    nonneg = Float32[max(w, 0.0f0) for w in weights]
    total = sum(nonneg)
    if total <= 0.0f0
        return Float32[1 / 3, 1 / 3, 1 / 3]
    end
    return Float32[w / total for w in nonneg]
end

function get_ensemble_weights(farm_id::String)::Vector{Float32}
    weights = get(ENSEMBLE_WEIGHT_CACHE, farm_id, Float32[1 / 3, 1 / 3, 1 / 3])
    if length(weights) != 3
        return Float32[1 / 3, 1 / 3, 1 / 3]
    end
    return _normalise_weights(weights)
end

function set_ensemble_weights!(farm_id::String, weights::Vector{Float32})::Vector{Float32}
    normalised = _normalise_weights(weights)
    ENSEMBLE_WEIGHT_CACHE[farm_id] = normalised
    return normalised
end

function evict_ensemble_weights!(farm_id::String)::Bool
    had = haskey(ENSEMBLE_WEIGHT_CACHE, farm_id)
    delete!(ENSEMBLE_WEIGHT_CACHE, farm_id)
    return had
end

function clear_ensemble_weights!()::Nothing
    empty!(ENSEMBLE_WEIGHT_CACHE)
    return nothing
end

function get_ensemble_hyperparams(farm_id::String)::Dict{String,Float32}
    defaults = Dict{String,Float32}(
        "exp_alpha" => 0.30f0,
        "exp_beta" => 0.10f0,
        "quantile_lambda" => 1.0f-3,
    )
    existing = get(ENSEMBLE_HYPERPARAM_CACHE, farm_id, defaults)
    merged = copy(defaults)
    for (k, v) in existing
        merged[k] = v
    end
    return merged
end

function set_ensemble_hyperparams!(
    farm_id::String,
    params::Dict{String,Float32},
)::Dict{String,Float32}
    merged = get_ensemble_hyperparams(farm_id)
    for (k, v) in params
        merged[k] = v
    end
    ENSEMBLE_HYPERPARAM_CACHE[farm_id] = merged
    return merged
end

function evict_ensemble_hyperparams!(farm_id::String)::Bool
    had = haskey(ENSEMBLE_HYPERPARAM_CACHE, farm_id)
    delete!(ENSEMBLE_HYPERPARAM_CACHE, farm_id)
    return had
end

function clear_ensemble_hyperparams!()::Nothing
    empty!(ENSEMBLE_HYPERPARAM_CACHE)
    return nothing
end

"""
    seasonal_decompose(history, period) -> (trend, seasonal, residual)

Simple additive decomposition using centered moving-average trend and a
phase-aligned seasonal component, with NaN-aware averaging.
"""
function seasonal_decompose(
    history::Vector{Float32},
    period::Int,
)::Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}
    n = length(history)
    n == 0 && return Float32[], Float32[], Float32[]

    use_period = clamp(period, 2, max(2, n))
    half_window = max(1, fld(use_period, 2))

    trend = Vector{Float32}(undef, n)
    for i in 1:n
        lo = max(1, i - half_window)
        hi = min(n, i + half_window)
        vals = Float32[]
        for v in @view history[lo:hi]
            if !isnan(v)
                push!(vals, v)
            end
        end
        trend[i] = isempty(vals) ? NaN32 : Float32(mean(vals))
    end

    seasonal_pattern = zeros(Float32, use_period)
    seasonal_counts = zeros(Int, use_period)
    for i in 1:n
        h = history[i]
        t = trend[i]
        if isnan(h) || isnan(t)
            continue
        end
        slot = mod1(i, use_period)
        seasonal_pattern[slot] += (h - t)
        seasonal_counts[slot] += 1
    end
    for i in 1:use_period
        if seasonal_counts[i] > 0
            seasonal_pattern[i] /= seasonal_counts[i]
        end
    end

    seasonal = Vector{Float32}(undef, n)
    residual = Vector{Float32}(undef, n)
    for i in 1:n
        seasonal[i] = seasonal_pattern[mod1(i, use_period)]
        h = history[i]
        t = trend[i]
        residual[i] = (isnan(h) || isnan(t)) ? NaN32 : (h - t - seasonal[i])
    end

    return trend, seasonal, residual
end

function _holt_linear_forecast(
    series::Vector{Float32};
    α::Float32=0.3f0,
    β::Float32=0.1f0,
)::Tuple{Float32, Float32}
    valid = Float32[v for v in series if !isnan(v)]
    if isempty(valid)
        return 0.0f0, 0.0f0
    end
    if length(valid) == 1
        return valid[1], max(abs(valid[1]) * 0.05f0, 1.0f-4)
    end

    level = valid[1]
    trend = valid[2] - valid[1]
    residuals = Float32[]

    for i in 2:length(valid)
        y = valid[i]
        pred = level + trend
        push!(residuals, y - pred)
        new_level = α * y + (1.0f0 - α) * (level + trend)
        new_trend = β * (new_level - level) + (1.0f0 - β) * trend
        level = new_level
        trend = new_trend
    end

    sigma = isempty(residuals) ? 0.0f0 : Float32(std(residuals; corrected=false))
    sigma = isfinite(sigma) ? max(sigma, 1.0f-4) : 1.0f-4
    return level + trend, sigma
end

function _aggregate_by_crop_bed(
    graph::LayeredHyperGraph,
    values::Vector{Float32},
)::Tuple{Vector{String}, Vector{Float32}}
    crop_layer = graph.layers[:crop_requirements]
    B_cpu = ensure_cpu(crop_layer.incidence)
    ne = size(B_cpu, 2)
    bed_ids = String[]
    bed_values = Float32[]
    for e in 1:ne
        members = findall(!iszero, @view B_cpu[:, e])
        isempty(members) && continue
        bed_id = e <= length(crop_layer.edge_ids) ? crop_layer.edge_ids[e] : "bed_$e"
        push!(bed_ids, bed_id)
        push!(bed_values, Float32(mean(values[members])))
    end
    return bed_ids, bed_values
end

function _weighted_quantile(
    values::Vector{Float32},
    weights::Vector{Float32},
    q::Float32,
)::Float32
    n = min(length(values), length(weights))
    n == 0 && return 0.0f0

    pairs = Tuple{Float32,Float32}[]
    for i in 1:n
        v = values[i]
        w = weights[i]
        if isfinite(v) && isfinite(w) && w > 0.0f0
            push!(pairs, (v, w))
        end
    end
    isempty(pairs) && return 0.0f0

    sort!(pairs; by=first)
    total_w = sum(last, pairs)
    total_w <= 0.0f0 && return first(pairs)[1]

    target = clamp(q, 0.0f0, 1.0f0) * total_w
    running = 0.0f0
    for (v, w) in pairs
        running += w
        if running >= target
            return v
        end
    end
    return pairs[end][1]
end

function _weighted_quantile_ci(
    member_lowers::Vector{Float32},
    member_medians::Vector{Float32},
    member_uppers::Vector{Float32},
    member_weights::Vector{Float32},
)::Tuple{Float32, Float32, Float32}
    n = min(
        length(member_lowers),
        length(member_medians),
        length(member_uppers),
        length(member_weights),
    )
    n == 0 && return 0.0f0, 0.0f0, 0.0f0

    # Approximate each member predictive distribution with (q10, q50, q90)
    # support points and aggregate via weighted quantiles of the mixture.
    support_values = Float32[]
    support_weights = Float32[]
    anchor_mass = Float32[0.20f0, 0.60f0, 0.20f0]

    for i in 1:n
        mw = max(member_weights[i], 0.0f0)
        mw <= 0.0f0 && continue
        lo = max(0.0f0, member_lowers[i])
        mid = max(lo, member_medians[i])
        hi = max(mid, member_uppers[i])
        append!(support_values, Float32[lo, mid, hi])
        append!(support_weights, Float32[
            mw * anchor_mass[1],
            mw * anchor_mass[2],
            mw * anchor_mass[3],
        ])
    end

    if isempty(support_values)
        return 0.0f0, 0.0f0, 0.0f0
    end

    q10 = _weighted_quantile(support_values, support_weights, 0.10f0)
    q50 = _weighted_quantile(support_values, support_weights, 0.50f0)
    q90 = _weighted_quantile(support_values, support_weights, 0.90f0)
    lo = max(0.0f0, min(q10, q50, q90))
    mid = max(lo, q50)
    hi = max(mid, q90)
    return mid, lo, hi
end

function _yield_feature_matrix(graph::LayeredHyperGraph)::Matrix{Float32}
    nv = graph.n_vertices
    feature_layers = Symbol[]
    for lsym in [:soil, :lighting, :crop_requirements, :vision]
        haskey(graph.layers, lsym) && push!(feature_layers, lsym)
    end

    X_raw = isempty(feature_layers) ? zeros(Float32, nv, 0) :
            Float32.(ensure_cpu(multi_layer_features(graph, feature_layers)))
    X_derived = Float32.(ensure_cpu(compute_derived_features(graph)))

    X = if size(X_derived, 2) > 0
        hcat(X_raw, X_derived)
    else
        X_raw
    end
    if size(X, 2) == 0
        return ones(Float32, nv, 1)
    end
    return X
end

function _compute_fao_vertex_prediction(graph::LayeredHyperGraph)::Vector{Float32}
    nv = graph.n_vertices
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
    return Float32.(ensure_cpu(y_fao))
end

@kernel function quantile_weight_kernel!(weights, residuals, q)
    i = @index(Global)
    @inbounds begin
        r = residuals[i]
        weights[i] = r >= 0.0f0 ? q : (1.0f0 - q)
    end
end

function _quantile_regression_irls(
    X::Matrix{Float32},
    y::Vector{Float32},
    q::Float32;
    λ::Float32=1.0f-3,
    iterations::Int=10,
)::Vector{Float32}
    n, p = size(X)
    yv = Float32[isfinite(v) ? v : 0.0f0 for v in y]
    Xv = hcat(ones(Float32, n), X)
    @inbounds for i in eachindex(Xv)
        if !isfinite(Xv[i])
            Xv[i] = 0.0f0
        end
    end
    β = zeros(Float32, p + 1)

    if HAS_CUDA
        Xg = CuArray(Xv)
        yg = CuArray(yv)
        βg = CuArray(β)
        backend = CUDABackend()
        nrows = size(Xg, 1)
        for _ in 1:iterations
            predg = Xg * βg
            rg = yg .- predg
            wg = similar(rg)
            launch_kernel!(quantile_weight_kernel!, backend, nrows, wg, rg, q)
            Wg = Diagonal(wg)
            Iregg = CuArray(Matrix{Float32}(I, p + 1, p + 1))
            lhsg = Xg' * Wg * Xg + λ * Iregg
            rhsg = Xg' * Wg * yg
            try
                βg = lhsg \ rhsg
            catch err
                if err isa LinearAlgebra.SingularException
                    lhs_cpu = ensure_cpu(lhsg)
                    rhs_cpu = ensure_cpu(rhsg)
                    β_cpu = pinv(lhs_cpu) * rhs_cpu
                    βg = CuArray(Float32.(β_cpu))
                else
                    rethrow(err)
                end
            end
        end
        return Float32.(ensure_cpu(βg))
    end

    for _ in 1:iterations
        pred = Xv * β
        r = yv .- pred
        w = similar(r)
        @inbounds for i in eachindex(r)
            ri = r[i]
            if !isfinite(ri)
                w[i] = 0.5f0
            else
                w[i] = ri >= 0.0f0 ? q : (1.0f0 - q)
            end
        end
        W = Diagonal(w)
        Ireg = Matrix{Float32}(I, p + 1, p + 1)
        lhs = Xv' * W * Xv + λ * Ireg
        rhs = Xv' * W * yv
        @inbounds for i in eachindex(lhs)
            if !isfinite(lhs[i])
                lhs[i] = 0.0f0
            end
        end
        @inbounds for i in eachindex(rhs)
            if !isfinite(rhs[i])
                rhs[i] = 0.0f0
            end
        end
        try
            β = lhs \ rhs
        catch err
            if err isa LinearAlgebra.SingularException || err isa ArgumentError
                β = pinv(lhs) * rhs
            else
                rethrow(err)
            end
        end
    end
    return β
end

"""
    compute_exp_smoothing_forecast(graph) -> (y_pred, y_lower, y_upper)

Compute per-vertex forecast from soil-moisture history using additive seasonal
decomposition + Holt linear smoothing on deseasonalized values.
"""
function compute_exp_smoothing_forecast(
    graph::LayeredHyperGraph,
    ;
    alpha::Float32=0.30f0,
    beta::Float32=0.10f0,
)::Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}
    nv = graph.n_vertices
    y_pred = zeros(Float32, nv)
    y_lower = zeros(Float32, nv)
    y_upper = zeros(Float32, nv)

    has_soil = haskey(graph.layers, :soil)
    y_pot = haskey(graph.layers, :crop_requirements) ?
            Float32.(ensure_cpu(graph.layers[:crop_requirements].vertex_features[:, 1])) :
            ones(Float32, nv)

    for v in 1:nv
        moisture_series = Float32[]
        if has_soil
            hist, mask = get_history(graph.layers[:soil], v; return_mask=true)
            if size(hist, 2) > 0
                for t in 1:size(hist, 2)
                    if mask[1, t]
                        push!(moisture_series, Float32(hist[1, t]))
                    end
                end
            end
        end
        if isempty(moisture_series)
            if has_soil
                push!(moisture_series, Float32(graph.layers[:soil].vertex_features[v, 1]))
            else
                push!(moisture_series, 0.25f0)
            end
        end

        use_period = min(96, max(2, length(moisture_series)))
        _, seasonal, residual = seasonal_decompose(moisture_series, use_period)
        deseason = similar(moisture_series)
        for i in eachindex(moisture_series)
            m = moisture_series[i]
            s = seasonal[i]
            deseason[i] = (isnan(m) || isnan(s)) ? m : (m - s)
        end

        forecast_moisture, sigma = _holt_linear_forecast(deseason; α=alpha, β=beta)
        moisture_health = clamp(
            (forecast_moisture - DEFAULT_WILTING_POINT) /
            (DEFAULT_FIELD_CAPACITY - DEFAULT_WILTING_POINT),
            0.0f0,
            1.0f0,
        )

        pred = y_pot[v] * moisture_health
        residual_sigma = Float32(std(Float32[r for r in residual if !isnan(r)]; corrected=false))
        residual_sigma = isfinite(residual_sigma) ? residual_sigma : 0.0f0
        half = max(0.10f0 * abs(pred), 1.96f0 * (sigma + residual_sigma) * abs(y_pot[v]))

        y_pred[v] = pred
        y_lower[v] = max(0.0f0, pred - half)
        y_upper[v] = pred + half
    end

    return y_pred, y_lower, y_upper
end

"""
    compute_quantile_regression_forecast(graph; quantiles=[0.10, 0.50, 0.90])
        -> (y_median, y_q10, y_q90)

Quantile regression on the existing multi-layer feature matrix.
"""
function compute_quantile_regression_forecast(
    graph::LayeredHyperGraph;
    quantiles::Vector{Float32}=Float32[0.10f0, 0.50f0, 0.90f0],
    lambda::Float32=1.0f-3,
)::Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}
    X = _yield_feature_matrix(graph)
    y_target = _compute_fao_vertex_prediction(graph)

    q10 = clamp(quantiles[1], 0.01f0, 0.49f0)
    q50 = clamp(quantiles[2], 0.10f0, 0.90f0)
    q90 = clamp(quantiles[3], 0.51f0, 0.99f0)

    β10 = _quantile_regression_irls(X, y_target, q10; λ=lambda)
    β50 = _quantile_regression_irls(X, y_target, q50; λ=lambda)
    β90 = _quantile_regression_irls(X, y_target, q90; λ=lambda)

    Xv = hcat(ones(Float32, size(X, 1)), X)
    pred10 = Xv * β10
    pred50 = Xv * β50
    pred90 = Xv * β90

    # Enforce monotonic quantiles per vertex.
    for i in eachindex(pred50)
        lo = min(pred10[i], pred50[i], pred90[i])
        mid = pred50[i]
        hi = max(pred10[i], pred50[i], pred90[i])
        pred10[i] = max(0.0f0, lo)
        pred50[i] = max(pred10[i], max(0.0f0, mid))
        pred90[i] = max(pred50[i], max(0.0f0, hi))
    end

    return Float32.(pred50), Float32.(pred10), Float32.(pred90)
end

"""
    compute_ensemble_yield_forecast(graph; include_members=false) -> Vector{Dict}

Blend FAO single-model, exponential-smoothing, and quantile-regression members
with per-farm cached weights.
"""
function compute_ensemble_yield_forecast(
    graph::LayeredHyperGraph;
    include_members::Bool=false,
)::Vector{Dict{String,Any}}
    haskey(graph.layers, :crop_requirements) || return Dict{String,Any}[]

    farm_id = graph.farm_id
    weights = get_ensemble_weights(farm_id)
    hyperparams = get_ensemble_hyperparams(farm_id)
    exp_alpha = get(hyperparams, "exp_alpha", 0.30f0)
    exp_beta = get(hyperparams, "exp_beta", 0.10f0)
    quantile_lambda = get(hyperparams, "quantile_lambda", 1.0f-3)

    single_results = compute_yield_forecast_single(graph)
    y_exp, y_exp_lo, y_exp_hi = compute_exp_smoothing_forecast(
        graph;
        alpha=exp_alpha,
        beta=exp_beta,
    )
    y_q50, y_q10, y_q90 = compute_quantile_regression_forecast(
        graph;
        lambda=quantile_lambda,
    )

    bed_ids, exp_vals = _aggregate_by_crop_bed(graph, y_exp)
    _, exp_lowers = _aggregate_by_crop_bed(graph, y_exp_lo)
    _, exp_uppers = _aggregate_by_crop_bed(graph, y_exp_hi)
    _, q50_vals = _aggregate_by_crop_bed(graph, y_q50)
    _, q10_vals = _aggregate_by_crop_bed(graph, y_q10)
    _, q90_vals = _aggregate_by_crop_bed(graph, y_q90)

    n = min(length(single_results), length(exp_vals), length(q50_vals))
    results = Dict{String,Any}[]
    for i in 1:n
        single = single_results[i]
        fao_est = Float32(single["yield_estimate_kg_m2"])
        fao_lo = Float32(single["yield_lower"])
        fao_hi = Float32(single["yield_upper"])

        exp_est = exp_vals[i]
        exp_lo = exp_lowers[i]
        exp_hi = exp_uppers[i]

        qr_est = q50_vals[i]
        qr_lo = q10_vals[i]
        qr_hi = q90_vals[i]

        ens_est, ens_lo, ens_hi = _weighted_quantile_ci(
            Float32[fao_lo, exp_lo, qr_lo],
            Float32[fao_est, exp_est, qr_est],
            Float32[fao_hi, exp_hi, qr_hi],
            weights,
        )

        item = Dict{String,Any}(
            "crop_bed_id" => bed_ids[i],
            "yield_estimate_kg_m2" => Float64(ens_est),
            "yield_lower" => Float64(ens_lo),
            "yield_upper" => Float64(ens_hi),
            "confidence" => 0.90,
            "stress_factors" => get(single, "stress_factors", Dict{String,Any}()),
            "model_layer" => "ensemble",
            "ensemble_weights" => Dict{String,Any}(
                "fao_single" => Float64(weights[1]),
                "exp_smoothing" => Float64(weights[2]),
                "quantile_regression" => Float64(weights[3]),
            ),
            "hyperparameters" => Dict{String,Any}(
                "exp_alpha" => Float64(exp_alpha),
                "exp_beta" => Float64(exp_beta),
                "quantile_lambda" => Float64(quantile_lambda),
            ),
        )

        if include_members
            item["ensemble_members"] = Vector{Dict{String,Any}}([
                Dict{String,Any}(
                    "model_name" => "fao_single",
                    "yield_estimate" => Float64(fao_est),
                    "lower" => Float64(fao_lo),
                    "upper" => Float64(fao_hi),
                    "weight" => Float64(weights[1]),
                ),
                Dict{String,Any}(
                    "model_name" => "exp_smoothing",
                    "yield_estimate" => Float64(exp_est),
                    "lower" => Float64(exp_lo),
                    "upper" => Float64(exp_hi),
                    "weight" => Float64(weights[2]),
                ),
                Dict{String,Any}(
                    "model_name" => "quantile_regression",
                    "yield_estimate" => Float64(qr_est),
                    "lower" => Float64(qr_lo),
                    "upper" => Float64(qr_hi),
                    "weight" => Float64(weights[3]),
                ),
            ])
        end

        push!(results, item)
    end

    return results
end
