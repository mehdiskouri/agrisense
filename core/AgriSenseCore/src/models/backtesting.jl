# ---------------------------------------------------------------------------
# backtesting.jl — strict temporal backtesting on synthetic-yield oracle
# ---------------------------------------------------------------------------

function _fold_train_windows(total_steps::Int, n_folds::Int, min_history::Int)::Vector{Int}
    if total_steps <= min_history + 1
        return Int[]
    end
    folds = max(1, n_folds)
    usable = total_steps - min_history
    step = max(1, fld(usable, folds + 1))
    windows = Int[]
    cursor = min_history
    for _ in 1:folds
        cursor = min(cursor + step, total_steps - 1)
        push!(windows, cursor)
    end
    return unique(windows)
end

function _safe_mean(values::Vector{Float32})::Float32
    finite = Float32[v for v in values if isfinite(v)]
    isempty(finite) && return 0.0f0
    return Float32(mean(finite))
end

function _mae(y_true::Vector{Float32}, y_pred::Vector{Float32})::Float32
    n = min(length(y_true), length(y_pred))
    n == 0 && return 0.0f0
    errs = Float32[]
    for i in 1:n
        yt = y_true[i]
        yp = y_pred[i]
        if isfinite(yt) && isfinite(yp)
            push!(errs, abs(yt - yp))
        end
    end
    return _safe_mean(errs)
end

function _rmse(y_true::Vector{Float32}, y_pred::Vector{Float32})::Float32
    n = min(length(y_true), length(y_pred))
    n == 0 && return 0.0f0
    sq = Float32[]
    for i in 1:n
        yt = y_true[i]
        yp = y_pred[i]
        if isfinite(yt) && isfinite(yp)
            d = yt - yp
            push!(sq, d * d)
        end
    end
    return Float32(sqrt(_safe_mean(sq)))
end

function _coverage(y_true::Vector{Float32}, y_lo::Vector{Float32}, y_hi::Vector{Float32})::Float32
    n = min(length(y_true), length(y_lo), length(y_hi))
    n == 0 && return 0.0f0
    hit = 0
    denom = 0
    for i in 1:n
        yt = y_true[i]
        lo = y_lo[i]
        hi = y_hi[i]
        if !(isfinite(yt) && isfinite(lo) && isfinite(hi))
            continue
        end
        denom += 1
        if yt >= lo && yt <= hi
            hit += 1
        end
    end
    denom == 0 && return 0.0f0
    return Float32(hit / denom)
end

function _inverse_mae_weights(maes::Tuple{Float32, Float32, Float32})::Vector{Float32}
    eps = 1.0f-4
    a = isfinite(maes[1]) ? maes[1] : 1.0f3
    b = isfinite(maes[2]) ? maes[2] : 1.0f3
    c = isfinite(maes[3]) ? maes[3] : 1.0f3
    raw = Float32[1.0f0 / (a + eps), 1.0f0 / (b + eps), 1.0f0 / (c + eps)]
    return _normalise_weights(raw)
end

function _history_feature_at(
    layer::HyperGraphLayer,
    vertex_idx::Integer,
    feature_idx::Integer,
    step_idx::Integer,
    default::Float32,
)::Float32
    data, mask = get_history(layer, Int(vertex_idx); return_mask=true)
    if size(data, 2) == 0 || feature_idx > size(data, 1)
        return default
    end
    idx = clamp(step_idx, 1, size(data, 2))
    return mask[feature_idx, idx] ? Float32(data[feature_idx, idx]) : default
end

function _temperature_stress(temp::Float32)::Float32
    if temp < 5.0f0
        return 0.0f0
    elseif temp < 15.0f0
        return (temp - 5.0f0) / 10.0f0
    elseif temp <= 30.0f0
        return 1.0f0
    elseif temp < 40.0f0
        return (40.0f0 - temp) / 10.0f0
    end
    return 0.0f0
end

function _oracle_window_actuals(
    graph::LayeredHyperGraph,
    start_step::Int,
    end_step::Int,
    bed_ids::Vector{String},
)::Vector{Float32}
    if !haskey(graph.layers, :crop_requirements) || start_step > end_step
        return zeros(Float32, length(bed_ids))
    end

    crop_layer = graph.layers[:crop_requirements]
    B_cpu = ensure_cpu(crop_layer.incidence)
    ne = size(B_cpu, 2)
    bed_to_edge = Dict{String,Int}()
    for e in 1:ne
        bed_id = e <= length(crop_layer.edge_ids) ? crop_layer.edge_ids[e] : "bed_$e"
        bed_to_edge[bed_id] = e
    end

    actual = zeros(Float32, length(bed_ids))
    t0 = max(1, start_step)
    t1 = max(t0, end_step)

    Threads.@threads for i in eachindex(bed_ids)
        bed_id = bed_ids[i]
        edge_idx = get(bed_to_edge, bed_id, 0)
        if edge_idx == 0
            actual[i] = 0.0f0
            continue
        end

        members = findall(!iszero, @view B_cpu[:, edge_idx])
        if isempty(members)
            actual[i] = 0.0f0
            continue
        end

        y_pot = Float32(mean(Float32.(ensure_cpu(crop_layer.vertex_features[members, 1]))))
        growth_progress = Float32(mean(Float32.(ensure_cpu(crop_layer.vertex_features[members, 2]))))
        growth_scale = clamp(0.35f0 + 0.65f0 * growth_progress, 0.2f0, 1.0f0)

        target_n = Float32(mean(Float32.(ensure_cpu(crop_layer.vertex_features[members, 3]))))
        target_p = Float32(mean(Float32.(ensure_cpu(crop_layer.vertex_features[members, 4]))))
        target_k = Float32(mean(Float32.(ensure_cpu(crop_layer.vertex_features[members, 5]))))

        samples = Float32[]
        for t in t0:t1
            soil_m = if haskey(graph.layers, :soil)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:soil], v, 1, t, 0.25f0))
                end
                _safe_mean(vals)
            else
                0.25f0
            end

            temp = if haskey(graph.layers, :weather)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:weather], v, 1, t, 22.0f0))
                end
                _safe_mean(vals)
            else
                22.0f0
            end

            precip = if haskey(graph.layers, :weather)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:weather], v, 3, t, 0.0f0))
                end
                _safe_mean(vals)
            else
                0.0f0
            end

            dli = if haskey(graph.layers, :lighting)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:lighting], v, 2, t, 20.0f0))
                end
                _safe_mean(vals)
            else
                20.0f0
            end

            anomaly = if haskey(graph.layers, :vision)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:vision], v, 3, t, 0.0f0))
                end
                _safe_mean(vals)
            else
                0.0f0
            end

            npk_n = if haskey(graph.layers, :npk)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:npk], v, 1, t, target_n))
                end
                _safe_mean(vals)
            else
                target_n
            end
            npk_p = if haskey(graph.layers, :npk)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:npk], v, 2, t, target_p))
                end
                _safe_mean(vals)
            else
                target_p
            end
            npk_k = if haskey(graph.layers, :npk)
                vals = Float32[]
                for v in members
                    push!(vals, _history_feature_at(graph.layers[:npk], v, 3, t, target_k))
                end
                _safe_mean(vals)
            else
                target_k
            end

            ks = clamp(
                (soil_m - DEFAULT_WILTING_POINT) /
                (DEFAULT_FIELD_CAPACITY - DEFAULT_WILTING_POINT),
                0.0f0,
                1.0f0,
            )

            d_n = target_n > 0.0f0 ? max(target_n - npk_n, 0.0f0) / target_n : 0.0f0
            d_p = target_p > 0.0f0 ? max(target_p - npk_p, 0.0f0) / target_p : 0.0f0
            d_k = target_k > 0.0f0 ? max(target_k - npk_k, 0.0f0) / target_k : 0.0f0
            kn = clamp(1.0f0 - (d_n + d_p + d_k) / 3.0f0, 0.0f0, 1.0f0)

            kw = _temperature_stress(temp)
            kw = clamp(kw * (1.0f0 + 0.05f0 * clamp(precip / 5.0f0, 0.0f0, 1.0f0)), 0.0f0, 1.0f0)

            kl = clamp(dli / 20.0f0, 0.0f0, 1.0f0)
            kv = clamp(1.0f0 - 0.5f0 * max(anomaly, 0.0f0), 0.70f0, 1.0f0)

            t_days = Float32(t) / 96.0f0
            seasonal = 1.0f0 + 0.08f0 * sin(Float32(2.0f0 * pi) * t_days / 30.0f0)
            quality = 0.35f0 * ks + 0.25f0 * kn + 0.20f0 * kw + 0.20f0 * kl

            oracle = max(0.0f0, y_pot * growth_scale * seasonal * quality * kv)
            push!(samples, oracle)
        end

        actual[i] = _safe_mean(samples)
    end

    return actual
end

function _snapshot_graph_at_step(
    graph::LayeredHyperGraph,
    step_idx::Int,
)::LayeredHyperGraph
    step = max(0, step_idx)
    layers = Dict{Symbol, HyperGraphLayer}()

    for (layer_name, layer) in graph.layers
        nv, d = size(layer.vertex_features)
        buf_size = size(layer.feature_history, 3)
        layer_len = min(step, layer.history_length)

        incidence = copy(ensure_cpu(layer.incidence))
        vertex_features = fill(Float32(NaN), nv, d)
        feature_history = zeros(Float32, nv, d, buf_size)
        feature_history_mask = fill(false, nv, d, buf_size)
        vertex_features_mask = fill(false, nv, d)

        for v in 1:nv
            hist, hmask = get_history(layer, v; return_mask=true)
            available = min(layer_len, size(hist, 2))
            if available > 0
                feature_history[v, :, 1:available] .= hist[:, 1:available]
                feature_history_mask[v, :, 1:available] .= hmask[:, 1:available]
                for f in 1:d
                    if hmask[f, available]
                        vertex_features[v, f] = Float32(hist[f, available])
                        vertex_features_mask[v, f] = true
                    end
                end
            end
        end

        snapshot = HyperGraphLayer(
            incidence,
            vertex_features,
            feature_history,
            mod1(layer_len + 1, buf_size),
            layer_len,
            copy(layer.edge_metadata),
            copy(layer.vertex_ids),
            copy(layer.edge_ids),
        )
        snapshot.feature_history_mask = feature_history_mask
        snapshot.vertex_features_mask = vertex_features_mask
        layers[layer_name] = snapshot
    end

    return LayeredHyperGraph(graph.farm_id, graph.n_vertices, copy(graph.vertex_index), layers)
end

function _collect_member_predictions(
    graph::LayeredHyperGraph,
    alpha::Float32,
    beta::Float32,
    lambda::Float32,
)::Dict{String,Any}
    single_results = compute_yield_forecast_single(graph)

    y_exp, y_exp_lo, y_exp_hi = compute_exp_smoothing_forecast(
        graph;
        alpha=alpha,
        beta=beta,
    )
    y_q50, y_q10, y_q90 = compute_quantile_regression_forecast(
        graph;
        lambda=lambda,
    )

    bed_ids, exp_vals = _aggregate_by_crop_bed(graph, y_exp)
    _, exp_lowers = _aggregate_by_crop_bed(graph, y_exp_lo)
    _, exp_uppers = _aggregate_by_crop_bed(graph, y_exp_hi)
    _, q50_vals = _aggregate_by_crop_bed(graph, y_q50)
    _, q10_vals = _aggregate_by_crop_bed(graph, y_q10)
    _, q90_vals = _aggregate_by_crop_bed(graph, y_q90)

    n = min(length(single_results), length(exp_vals), length(q50_vals))
    fao_est = zeros(Float32, n)
    fao_lo = zeros(Float32, n)
    fao_hi = zeros(Float32, n)
    exp_est = zeros(Float32, n)
    exp_lo = zeros(Float32, n)
    exp_hi = zeros(Float32, n)
    qr_est = zeros(Float32, n)
    qr_lo = zeros(Float32, n)
    qr_hi = zeros(Float32, n)
    ids = String[]

    for i in 1:n
        push!(ids, bed_ids[i])
        s = single_results[i]
        fao_est[i] = Float32(s["yield_estimate_kg_m2"])
        fao_lo[i] = Float32(s["yield_lower"])
        fao_hi[i] = Float32(s["yield_upper"])

        exp_est[i] = exp_vals[i]
        exp_lo[i] = exp_lowers[i]
        exp_hi[i] = exp_uppers[i]

        qr_est[i] = q50_vals[i]
        qr_lo[i] = q10_vals[i]
        qr_hi[i] = q90_vals[i]
    end

    return Dict{String,Any}(
        "bed_ids" => ids,
        "fao_est" => fao_est,
        "fao_lo" => fao_lo,
        "fao_hi" => fao_hi,
        "exp_est" => exp_est,
        "exp_lo" => exp_lo,
        "exp_hi" => exp_hi,
        "qr_est" => qr_est,
        "qr_lo" => qr_lo,
        "qr_hi" => qr_hi,
    )
end

function _ensemble_from_members(
    members::Dict{String,Any},
    weights::Vector{Float32},
)::Tuple{Vector{Float32}, Vector{Float32}, Vector{Float32}}
    n = length(members["bed_ids"])
    ens = zeros(Float32, n)
    lo = zeros(Float32, n)
    hi = zeros(Float32, n)

    for i in 1:n
        mid, q10, q90 = _weighted_quantile_ci(
            Float32[members["fao_lo"][i], members["exp_lo"][i], members["qr_lo"][i]],
            Float32[members["fao_est"][i], members["exp_est"][i], members["qr_est"][i]],
            Float32[members["fao_hi"][i], members["exp_hi"][i], members["qr_hi"][i]],
            weights,
        )
        ens[i] = mid
        lo[i] = q10
        hi[i] = q90
    end

    return ens, lo, hi
end

function _evaluate_hyperparams(
    graph::LayeredHyperGraph,
    train_windows::Vector{Int},
    train_limit::Int,
    alpha::Float32,
    beta::Float32,
    lambda::Float32,
)::Tuple{Float32, Float32, Float32, Float32}
    mae_fao = Float32[]
    mae_exp = Float32[]
    mae_qr = Float32[]
    mae_ens = Float32[]

    for i in eachindex(train_windows)
        train_end = train_windows[i]
        eval_end = i < length(train_windows) ? train_windows[i + 1] : train_limit
        eval_end <= train_end && continue

        snapshot = _snapshot_graph_at_step(graph, train_end)
        members = _collect_member_predictions(snapshot, alpha, beta, lambda)
        bed_ids = members["bed_ids"]
        actual = _oracle_window_actuals(graph, train_end + 1, eval_end, bed_ids)

        n = min(length(actual), length(members["fao_est"]))
        n == 0 && continue
        push!(mae_fao, _mae(actual[1:n], members["fao_est"][1:n]))
        push!(mae_exp, _mae(actual[1:n], members["exp_est"][1:n]))
        push!(mae_qr, _mae(actual[1:n], members["qr_est"][1:n]))

        ens, _, _ = _ensemble_from_members(members, Float32[1 / 3, 1 / 3, 1 / 3])
        push!(mae_ens, _mae(actual[1:n], ens[1:n]))
    end

    return _safe_mean(mae_fao), _safe_mean(mae_exp), _safe_mean(mae_qr), _safe_mean(mae_ens)
end

"""
    optimize_ensemble_hyperparams!(graph; n_folds=5, min_history=24) -> Dict

Per-farm grid search over smoothing and quantile regularization hyperparameters.
Stores the best combination in `ENSEMBLE_HYPERPARAM_CACHE`.
"""
function optimize_ensemble_hyperparams!(
    graph::LayeredHyperGraph;
    n_folds::Int=5,
    min_history::Int=24,
)::Dict{String,Float32}
    total_steps = haskey(graph.layers, :soil) ? graph.layers[:soil].history_length : 0
    train_end = max(min_history, Int(floor(total_steps * 0.75)))
    train_windows = _fold_train_windows(train_end, n_folds, min_history)
    if isempty(train_windows)
        return set_ensemble_hyperparams!(graph.farm_id, Dict{String,Float32}(
            "exp_alpha" => 0.30f0,
            "exp_beta" => 0.10f0,
            "quantile_lambda" => 1.0f-3,
        ))
    end

    alpha_grid = Float32[0.20f0, 0.30f0, 0.45f0]
    beta_grid = Float32[0.05f0, 0.10f0, 0.20f0]
    lambda_grid = Float32[1.0f-4, 1.0f-3, 1.0f-2]

    best_score = typemax(Float32)
    best = Dict{String,Float32}(
        "exp_alpha" => 0.30f0,
        "exp_beta" => 0.10f0,
        "quantile_lambda" => 1.0f-3,
    )

    for alpha in alpha_grid
        for beta in beta_grid
            for lambda in lambda_grid
                _, _, _, ens_mae = _evaluate_hyperparams(
                    graph,
                    train_windows,
                    train_end,
                    alpha,
                    beta,
                    lambda,
                )
                if ens_mae < best_score
                    best_score = ens_mae
                    best["exp_alpha"] = alpha
                    best["exp_beta"] = beta
                    best["quantile_lambda"] = lambda
                end
            end
        end
    end

    return set_ensemble_hyperparams!(graph.farm_id, best)
end

"""
    backtest_yield_ensemble(graph; n_folds=5, min_history=24) -> Dict

Strict temporal split backtest:
  - train: months 1-9 (first 75% of available history)
  - validation: months 10-11 (next 16.7%)
  - test: month 12 (final 8.3%)

Within train, expanding-window folds are used to estimate member MAE and update
ensemble weights via inverse-MAE blending.
"""
function backtest_yield_ensemble(
    graph::LayeredHyperGraph;
    n_folds::Int=5,
    min_history::Int=24,
)::Dict{String,Any}
    haskey(graph.layers, :crop_requirements) || return Dict{String,Any}(
        "farm_id" => graph.farm_id,
        "n_folds" => 0,
        "per_fold_metrics" => Dict{String,Any}[],
        "aggregate_metrics" => Dict{String,Any}(),
        "weights" => Dict{String,Any}("fao_single" => 1 / 3, "exp_smoothing" => 1 / 3, "quantile_regression" => 1 / 3),
        "status" => "no_crop_layer",
    )

    total_steps = haskey(graph.layers, :soil) ? graph.layers[:soil].history_length : 0
    if total_steps < min_history + 12
        return Dict{String,Any}(
            "farm_id" => graph.farm_id,
            "n_folds" => 0,
            "per_fold_metrics" => Dict{String,Any}[],
            "aggregate_metrics" => Dict{String,Any}(),
            "weights" => Dict{String,Any}("fao_single" => 1 / 3, "exp_smoothing" => 1 / 3, "quantile_regression" => 1 / 3),
            "status" => "insufficient_history",
        )
    end

    train_end = max(min_history, Int(floor(total_steps * 0.75)))
    val_end = max(train_end + 1, Int(floor(total_steps * (11.0 / 12.0))))
    test_end = total_steps

    train_windows = _fold_train_windows(train_end, n_folds, min_history)
    folds = length(train_windows)

    tuned = optimize_ensemble_hyperparams!(graph; n_folds=n_folds, min_history=min_history)
    alpha = get(tuned, "exp_alpha", 0.30f0)
    beta = get(tuned, "exp_beta", 0.10f0)
    lambda = get(tuned, "quantile_lambda", 1.0f-3)

    fao_maes = Float32[]
    exp_maes = Float32[]
    qr_maes = Float32[]
    ens_maes = Float32[]
    fold_payload = Dict{String,Any}[]

    for i in eachindex(train_windows)
        fold_train_end = train_windows[i]
        fold_eval_end = i < length(train_windows) ? train_windows[i + 1] : train_end
        fold_eval_end <= fold_train_end && continue

        snapshot = _snapshot_graph_at_step(graph, fold_train_end)
        members = _collect_member_predictions(snapshot, alpha, beta, lambda)
        bed_ids = members["bed_ids"]
        actual = _oracle_window_actuals(graph, fold_train_end + 1, fold_eval_end, bed_ids)

        n = min(length(actual), length(members["fao_est"]))
        n == 0 && continue

        fao = members["fao_est"][1:n]
        exp = members["exp_est"][1:n]
        qr = members["qr_est"][1:n]
        ens, ens_lo, ens_hi = _ensemble_from_members(members, Float32[1 / 3, 1 / 3, 1 / 3])
        ens = ens[1:n]
        ens_lo = ens_lo[1:n]
        ens_hi = ens_hi[1:n]
        y_true = actual[1:n]

        fao_mae = _mae(y_true, fao)
        exp_mae = _mae(y_true, exp)
        qr_mae = _mae(y_true, qr)
        ens_mae = _mae(y_true, ens)

        push!(fao_maes, fao_mae)
        push!(exp_maes, exp_mae)
        push!(qr_maes, qr_mae)
        push!(ens_maes, ens_mae)

        push!(fold_payload, Dict{String,Any}(
            "fold" => i,
            "train_window" => fold_train_end,
            "eval_window" => Dict{String,Any}("start" => fold_train_end + 1, "end" => fold_eval_end),
            "metrics" => Dict{String,Any}(
                "fao_single" => Dict("mae" => Float64(fao_mae), "rmse" => Float64(_rmse(y_true, fao))),
                "exp_smoothing" => Dict("mae" => Float64(exp_mae), "rmse" => Float64(_rmse(y_true, exp))),
                "quantile_regression" => Dict("mae" => Float64(qr_mae), "rmse" => Float64(_rmse(y_true, qr))),
                "ensemble" => Dict(
                    "mae" => Float64(ens_mae),
                    "rmse" => Float64(_rmse(y_true, ens)),
                    "ci_coverage" => Float64(_coverage(y_true, ens_lo, ens_hi)),
                ),
            ),
        ))
    end

    agg_fao = _safe_mean(fao_maes)
    agg_exp = _safe_mean(exp_maes)
    agg_qr = _safe_mean(qr_maes)
    agg_ens = _safe_mean(ens_maes)

    updated_weights = _inverse_mae_weights((agg_fao, agg_exp, agg_qr))
    set_ensemble_weights!(graph.farm_id, updated_weights)

    val_snapshot = _snapshot_graph_at_step(graph, train_end)
    val_members = _collect_member_predictions(val_snapshot, alpha, beta, lambda)
    val_actual = _oracle_window_actuals(graph, train_end + 1, val_end, val_members["bed_ids"])
    val_pred, val_lo, val_hi = _ensemble_from_members(val_members, updated_weights)

    test_snapshot = _snapshot_graph_at_step(graph, val_end)
    test_members = _collect_member_predictions(test_snapshot, alpha, beta, lambda)
    test_actual = _oracle_window_actuals(graph, val_end + 1, test_end, test_members["bed_ids"])
    test_pred, test_lo, test_hi = _ensemble_from_members(test_members, updated_weights)

    n_val = min(length(val_actual), length(val_pred))
    n_test = min(length(test_actual), length(test_pred))

    return Dict{String,Any}(
        "farm_id" => graph.farm_id,
        "n_folds" => folds,
        "per_fold_metrics" => fold_payload,
        "aggregate_metrics" => Dict{String,Any}(
            "fao_single" => Dict("mae" => Float64(agg_fao)),
            "exp_smoothing" => Dict("mae" => Float64(agg_exp)),
            "quantile_regression" => Dict("mae" => Float64(agg_qr)),
            "ensemble" => Dict("mae" => Float64(agg_ens)),
            "validation" => Dict(
                "mae" => Float64(_mae(val_actual[1:n_val], val_pred[1:n_val])),
                "rmse" => Float64(_rmse(val_actual[1:n_val], val_pred[1:n_val])),
                "ci_coverage" => Float64(_coverage(val_actual[1:n_val], val_lo[1:n_val], val_hi[1:n_val])),
            ),
            "test" => Dict(
                "mae" => Float64(_mae(test_actual[1:n_test], test_pred[1:n_test])),
                "rmse" => Float64(_rmse(test_actual[1:n_test], test_pred[1:n_test])),
                "ci_coverage" => Float64(_coverage(test_actual[1:n_test], test_lo[1:n_test], test_hi[1:n_test])),
            ),
        ),
        "temporal_split" => Dict{String,Any}(
            "train" => Dict("start" => 1, "end" => train_end),
            "validation" => Dict("start" => train_end + 1, "end" => val_end),
            "test" => Dict("start" => val_end + 1, "end" => test_end),
        ),
        "oracle_provenance" => Dict{String,Any}(
            "type" => "synthetic_dgp",
            "target" => "yield_kg_m2",
            "source_layer" => "yield_oracle",
            "inputs" => ["soil", "weather", "npk", "lighting", "vision", "crop_requirements"],
            "policy" => "temporal_holdout_train_1_9_validate_10_11_test_12",
            "notes" => "Predictions are generated from fold snapshots without direct yield-column access",
        ),
        "hyperparameters" => Dict{String,Any}(
            "exp_alpha" => Float64(alpha),
            "exp_beta" => Float64(beta),
            "quantile_lambda" => Float64(lambda),
        ),
        "weights" => Dict{String,Any}(
            "fao_single" => Float64(updated_weights[1]),
            "exp_smoothing" => Float64(updated_weights[2]),
            "quantile_regression" => Float64(updated_weights[3]),
        ),
        "status" => "ok",
    )
end
