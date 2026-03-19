# ---------------------------------------------------------------------------
# Anomaly detection via statistical process control (GPU-first)
# ---------------------------------------------------------------------------

@kernel function rolling_stats_kernel!(means, stds, history, valid_length_arr, d)
    i = @index(Global)  # flat index over (vertex, feature)
    @inbounds begin
        v = div(i - 1, d) + 1
        f = mod(i - 1, d) + 1
        valid_len = valid_length_arr[1]
        buf_size = size(history, 3)

        if valid_len == 0
            means[v, f] = 0.0f0
            stds[v, f]  = 0.0f0
        else
            # Compute mean — skip NaN values (Phase 13)
            s = 0.0f0
            n_valid = Int32(0)
            for t in 1:valid_len
                idx = t <= buf_size ? t : mod1(t, buf_size)
                val = history[v, f, idx]
                if !isnan(val)
                    s += val
                    n_valid += Int32(1)
                end
            end
            if n_valid == Int32(0)
                means[v, f] = 0.0f0
                stds[v, f]  = 0.0f0
            else
                m = s / Float32(n_valid)
                means[v, f] = m

                # Compute std — skip NaN values
                ss = 0.0f0
                for t in 1:valid_len
                    idx = t <= buf_size ? t : mod1(t, buf_size)
                    val = history[v, f, idx]
                    if !isnan(val)
                        diff = val - m
                        ss += diff * diff
                    end
                end
                stds[v, f] = sqrt(ss / Float32(n_valid))
            end
        end
    end
end

@kernel function western_electric_kernel!(flags, current, means, stds,
                                          history, head_arr, valid_len_arr, buf_size,
                                          sigma1_map, sigma2_map, sigma3_map)
    i = @index(Global)  # flat index over (vertex, feature)
    @inbounds begin
        d = size(current, 2)
        v = div(i - 1, d) + 1
        f = mod(i - 1, d) + 1
        s = stds[v, f]
        m = means[v, f]
        sigma1 = sigma1_map[v, f]
        sigma2 = sigma2_map[v, f]
        sigma3 = sigma3_map[v, f]
        head = Int32(head_arr[1])
        valid_len = Int32(valid_len_arr[1])

        flag = Int32(0)  # bitfield: bit0=3σ, bit1=2of3>2σ, bit2=4of5>1σ, bit3=8consec

        cur = current[v, f]
        # Skip if current value is NaN or std is degenerate
        if s >= 1.0f-8 && !isnan(cur) && !isnan(m)
            deviation = abs(cur - m)

            # --- Rule 1: single point > 3σ ---
            if deviation > sigma3 * s
                flag = flag | Int32(1)  # bit 0
            end

            # Ring buffer: head points to NEXT write slot, so most recent is head-1

            # --- Rule 2: 2 of last 3 > 2σ (skip NaN history values) ---
            if valid_len >= 2
                count_2sigma = Int32(deviation > sigma2 * s ? 1 : 0)
                for k in 1:min(Int32(2), valid_len)
                    idx = mod1(head - k, buf_size)
                    hval = history[v, f, idx]
                    if !isnan(hval) && abs(hval - m) > sigma2 * s
                        count_2sigma += Int32(1)
                    end
                end
                if count_2sigma >= Int32(2)
                    flag = flag | Int32(2)  # bit 1
                end
            end

            # --- Rule 3: 4 of last 5 > 1σ (skip NaN history values) ---
            if valid_len >= 4
                count_1sigma = Int32(deviation > sigma1 * s ? 1 : 0)
                for k in 1:min(Int32(4), valid_len)
                    idx = mod1(head - k, buf_size)
                    hval = history[v, f, idx]
                    if !isnan(hval) && abs(hval - m) > sigma1 * s
                        count_1sigma += Int32(1)
                    end
                end
                if count_1sigma >= Int32(4)
                    flag = flag | Int32(4)  # bit 2
                end
            end

            # --- Rule 4: 8 consecutive same side of mean (NaN breaks the run) ---
            if valid_len >= 7
                cur_side = Int32(cur > m ? 1 : -1)
                all_same = true
                for k in 1:min(Int32(7), valid_len)
                    idx = mod1(head - k, buf_size)
                    hval = history[v, f, idx]
                    if isnan(hval)
                        all_same = false
                        break
                    end
                    hside = Int32(hval > m ? 1 : -1)
                    if hside != cur_side
                        all_same = false
                        break
                    end
                end
                if all_same
                    flag = flag | Int32(8)  # bit 3
                end
            end
        end  # s >= 1.0f-8 && !isnan(cur)

        flags[v, f] = flag
    end
end

@kernel function nan_run_kernel!(nan_runs, current, history, head_arr, valid_len_arr, buf_size, d)
    i = @index(Global)
    @inbounds begin
        v = div(i - 1, d) + 1
        f = mod(i - 1, d) + 1
        head = Int32(head_arr[1])
        valid_len = Int32(valid_len_arr[1])

        run_len = Int32(0)
        # Check current value (vertex_features) first — most recent reading
        if isnan(current[v, f])
            run_len = Int32(1)
            # Walk history backwards from head-1 (cap at buf_size to avoid wrap)
            max_k = min(valid_len, buf_size)
            for k in 1:max_k
                idx = mod1(head - k, buf_size)
                val = history[v, f, idx]
                if isnan(val)
                    run_len += Int32(1)
                else
                    break
                end
            end
        end
        nan_runs[v, f] = run_len
    end
end

# ---------------------------------------------------------------------------
# Minimum history required for Western Electric rules
# ---------------------------------------------------------------------------
const MIN_HISTORY_FOR_ANOMALY = 8
const MIN_NAN_RUN_FOR_OUTAGE = 4
const OUTAGE_ANOMALY_TYPE = "sensor_outage"
const DEFAULT_VISION_ANOMALY_SCORE_THRESHOLD = 0.7f0

@inline _to_float32(value, fallback::Float32)::Float32 =
    value isa Number ? Float32(value) : fallback

@inline _to_int(value, fallback::Int)::Int =
    value isa Integer ? Int(value) : fallback

@inline _to_bool(value, fallback::Bool)::Bool =
    value isa Bool ? value : fallback

function _resolve_threshold(
    threshold_cfg::Dict{String,Any},
    vertex_id::String,
    layer_name::String,
)::Dict{String,Any}
    defaults = get(threshold_cfg, "default", Dict{String,Any}())
    by_vertex_layer = get(threshold_cfg, "by_vertex_layer", Dict{String,Any}())
    by_layer = get(threshold_cfg, "by_layer", Dict{String,Any}())

    merged = Dict{String,Any}()
    if defaults isa Dict
        merge!(merged, defaults)
    end

    wildcard_key = string(vertex_id, "|*")
    exact_key = string(vertex_id, "|", layer_name)

    wildcard_cfg = get(by_vertex_layer, wildcard_key, nothing)
    if wildcard_cfg isa Dict
        merge!(merged, wildcard_cfg)
    end

    layer_cfg = get(by_layer, layer_name, nothing)
    if layer_cfg isa Dict
        merge!(merged, layer_cfg)
    end

    exact_cfg = get(by_vertex_layer, exact_key, nothing)
    if exact_cfg isa Dict
        merge!(merged, exact_cfg)
    end

    return merged
end

"""
    anomaly_type_from_layer(layer_sym::Symbol) -> String

Map layer name to anomaly category string.
"""
function anomaly_type_from_layer(layer_sym::Symbol)::String
    layer_sym == :soil    && return "environmental"
    layer_sym == :weather && return "environmental"
    layer_sym == :npk     && return "nutrient_imbalance"
    layer_sym == :vision  && return "visual_anomaly"
    layer_sym == :lighting && return "light_anomaly"
    layer_sym == :irrigation && return "irrigation_fault"
    return "unknown"
end

"""
    count_nan_run(layer::HyperGraphLayer, vertex_idx::Int) -> Vector{Int}

CPU helper: count consecutive NaN values from most recent slot backwards
in the ring buffer for each feature of the given vertex. Returns a vector
of length d (number of features). Mirrors `nan_run_kernel!` logic exactly.
"""
function count_nan_run(layer::HyperGraphLayer, vertex_idx::Int)::Vector{Int}
    d = size(layer.vertex_features, 2)
    buf_size = size(layer.feature_history, 3)
    # history_head points to the NEXT write slot; most-recent write is head-1
    head = mod1(layer.history_head - 1, buf_size)
    valid_len = layer.history_length
    history_cpu = ensure_cpu(layer.feature_history)
    vf_cpu = ensure_cpu(layer.vertex_features)

    runs = zeros(Int, d)
    for f in 1:d
        # Check current value first (vertex_features)
        if !isnan(vf_cpu[vertex_idx, f])
            continue  # Current is valid — no NaN run
        end
        runs[f] = 1  # Current is NaN
        # Walk history backwards (cap at buf_size to avoid wrap)
        for k in 1:min(valid_len, buf_size)
            idx = mod1(head - k, buf_size)
            if isnan(history_cpu[vertex_idx, f, idx])
                runs[f] += 1
            else
                break
            end
        end
    end
    return runs
end

"""
    compute_anomaly_detection(graph::LayeredHyperGraph) -> Vector{Dict}

Detect anomalies using rolling mean/σ and Western Electric rules on sensor streams.
GPU-accelerated: kernels compute on device, results pulled to CPU for Dict output.
"""
function compute_anomaly_detection(
    graph::LayeredHyperGraph;
    thresholds::Dict{String,Any}=Dict{String,Any}(),
)::Vector{Dict{String,Any}}
    results = Dict{String,Any}[]
    nv = graph.n_vertices

    # Track which vertices have anomalies per layer for cross-layer correlation
    soil_anomaly_vertices = Set{Int}()
    vision_anomaly_vertices = Set{Int}()
    outage_vertices = Dict{Symbol, Set{Int}}()

    # Feature names per layer
    feature_names = Dict{Symbol, Vector{String}}(
        :soil => ["moisture", "temperature", "conductivity", "pH"],
        :irrigation => ["flow_rate", "pressure", "valve_state"],
        :weather => ["temp", "humidity", "precip", "wind_speed", "solar_rad"],
        :npk => ["nitrogen", "phosphorus", "potassium"],
        :lighting => ["par", "dli", "spectrum_index"],
        :vision => ["canopy_coverage", "growth_stage", "anomaly_score", "ndvi"],
        :crop_requirements => ["target_yield", "growth_progress", "n_target", "p_target", "k_target"],
    )

    default_sigma1 = _to_float32(get(get(thresholds, "default", Dict{String,Any}()), "sigma1", 1.0f0), 1.0f0)
    default_sigma2 = _to_float32(get(get(thresholds, "default", Dict{String,Any}()), "sigma2", 2.0f0), 2.0f0)
    default_sigma3 = _to_float32(get(get(thresholds, "default", Dict{String,Any}()), "sigma3", 3.0f0), 3.0f0)
    default_min_history = _to_int(get(get(thresholds, "default", Dict{String,Any}()), "min_history", MIN_HISTORY_FOR_ANOMALY), MIN_HISTORY_FOR_ANOMALY)
    default_min_nan_run = _to_int(get(get(thresholds, "default", Dict{String,Any}()), "min_nan_run_outage", MIN_NAN_RUN_FOR_OUTAGE), MIN_NAN_RUN_FOR_OUTAGE)
    default_enabled = _to_bool(get(get(thresholds, "default", Dict{String,Any}()), "enabled", true), true)
    default_suppress_rule3_only = _to_bool(get(get(thresholds, "default", Dict{String,Any}()), "suppress_rule3_only", true), true)
    default_vision_score = _to_float32(get(get(thresholds, "default", Dict{String,Any}()), "vision_anomaly_score_threshold", DEFAULT_VISION_ANOMALY_SCORE_THRESHOLD), DEFAULT_VISION_ANOMALY_SCORE_THRESHOLD)

    # Rule name lookup for bitfield decoding
    rule_labels = ("3sigma", "2of3_2sigma", "4of5_1sigma", "8consec_same_side")

    for (layer_sym, layer) in graph.layers
        layer.history_length < MIN_HISTORY_FOR_ANOMALY && continue

        d = size(layer.vertex_features, 2)
        backend = array_backend(layer.vertex_features)
        ndrange = nv * d

        sigma1_cpu = fill(default_sigma1, nv, d)
        sigma2_cpu = fill(default_sigma2, nv, d)
        sigma3_cpu = fill(default_sigma3, nv, d)
        enabled_vertex = fill(default_enabled, nv)
        suppress_rule3_only_vertex = fill(default_suppress_rule3_only, nv)
        min_history_vertex = fill(default_min_history, nv)
        min_nan_run_vertex = fill(default_min_nan_run, nv)
        vision_score_vertex = fill(default_vision_score, nv)

        for v in 1:nv
            vid = v <= length(layer.vertex_ids) ? layer.vertex_ids[v] : string("v_", v)
            cfg = _resolve_threshold(thresholds, vid, string(layer_sym))
            sigma1_val = _to_float32(get(cfg, "sigma1", default_sigma1), default_sigma1)
            sigma2_val = _to_float32(get(cfg, "sigma2", default_sigma2), default_sigma2)
            sigma3_val = _to_float32(get(cfg, "sigma3", default_sigma3), default_sigma3)

            sigma1_cpu[v, :] .= sigma1_val
            sigma2_cpu[v, :] .= sigma2_val
            sigma3_cpu[v, :] .= sigma3_val
            enabled_vertex[v] = _to_bool(get(cfg, "enabled", default_enabled), default_enabled)
            suppress_rule3_only_vertex[v] = _to_bool(
                get(cfg, "suppress_rule3_only", default_suppress_rule3_only),
                default_suppress_rule3_only,
            )
            min_history_vertex[v] = _to_int(get(cfg, "min_history", default_min_history), default_min_history)
            min_nan_run_vertex[v] = _to_int(get(cfg, "min_nan_run_outage", default_min_nan_run), default_min_nan_run)
            vision_score_vertex[v] = _to_float32(
                get(cfg, "vision_anomaly_score_threshold", default_vision_score),
                default_vision_score,
            )
        end

        # Allocate on device
        means   = backend isa CPU ? zeros(Float32, nv, d) :
                  (HAS_CUDA ? CUDA.zeros(Float32, nv, d) : zeros(Float32, nv, d))
        stds    = backend isa CPU ? zeros(Float32, nv, d) :
                  (HAS_CUDA ? CUDA.zeros(Float32, nv, d) : zeros(Float32, nv, d))
        flags   = backend isa CPU ? zeros(Int32, nv, d) :
                  (HAS_CUDA ? CUDA.zeros(Int32, nv, d) : zeros(Int32, nv, d))
        current = layer.vertex_features   # already on device

        # Valid length as 1-element device array for kernel access
        valid_len_arr = backend isa CPU ? Int32[layer.history_length] :
                        (HAS_CUDA ? CuArray(Int32[layer.history_length]) :
                         Int32[layer.history_length])

        # Head pointer as 1-element device array
        # history_head points to the NEXT write slot; most-recent write is head-1
        head_val = Int32(mod1(layer.history_head - 1, size(layer.feature_history, 3)))
        head_arr = backend isa CPU ? Int32[head_val] :
                   (HAS_CUDA ? CuArray(Int32[head_val]) : Int32[head_val])
        buf_size = Int32(size(layer.feature_history, 3))

        # Rolling stats kernel
        launch_kernel!(rolling_stats_kernel!, backend, ndrange,
                       means, stds, layer.feature_history, valid_len_arr, Int32(d))

        sigma1_map = backend isa CPU ? sigma1_cpu :
                 (HAS_CUDA ? CuArray(sigma1_cpu) : sigma1_cpu)
        sigma2_map = backend isa CPU ? sigma2_cpu :
                 (HAS_CUDA ? CuArray(sigma2_cpu) : sigma2_cpu)
        sigma3_map = backend isa CPU ? sigma3_cpu :
                 (HAS_CUDA ? CuArray(sigma3_cpu) : sigma3_cpu)

        # Western Electric kernel (full 4-rule bitfield)
        launch_kernel!(western_electric_kernel!, backend, ndrange,
                       flags, current, means, stds,
                       layer.feature_history, head_arr, valid_len_arr, buf_size,
                   sigma1_map, sigma2_map, sigma3_map)

        # NaN-run kernel (contiguous outage detection)
        nan_runs = backend isa CPU ? zeros(Int32, nv, d) :
                   (HAS_CUDA ? CUDA.zeros(Int32, nv, d) : zeros(Int32, nv, d))
        launch_kernel!(nan_run_kernel!, backend, ndrange,
                       nan_runs, current, layer.feature_history, head_arr, valid_len_arr,
                       buf_size, Int32(d))

        # Pull results to CPU for Dict building
        flags_cpu    = ensure_cpu(flags)
        means_cpu    = ensure_cpu(means)
        stds_cpu     = ensure_cpu(stds)
        current_cpu  = ensure_cpu(current)
        nan_runs_cpu = ensure_cpu(nan_runs)

        # Collect flagged anomalies
        fnames = get(feature_names, layer_sym, ["feature_$i" for i in 1:d])

        # Timestamp range from cadence constant
        ts_end   = now(UTC)
        ts_start = ts_end - Minute(CADENCE_MINUTES * layer.history_length)

        for v in 1:nv, f in 1:d
            enabled_vertex[v] || continue
            layer.history_length >= min_history_vertex[v] || continue

            flags_cpu[v, f] == 0 && continue
            bitflag = flags_cpu[v, f]

            # Rule 3 (4-of-5 > 1σ) alone is intentionally treated as low-signal noise.
            # Require corroboration from stronger rules to avoid unstable baseline alerts.
            suppress_rule3_only_vertex[v] && bitflag == Int32(4) && continue

            vid = v <= length(layer.vertex_ids) ? layer.vertex_ids[v] : "v_$v"
            fname = f <= length(fnames) ? fnames[f] : "feature_$f"

            # Decode bitfield → severity + rule list
            fired_rules = String[]
            for bit in 0:3
                if (bitflag >> bit) & Int32(1) == Int32(1)
                    push!(fired_rules, rule_labels[bit + 1])
                end
            end
            severity = (bitflag & Int32(1)) != 0 ? "alarm" : "warning"

            sigma_dev = stds_cpu[v, f] > 1.0f-8 ?
                        abs(current_cpu[v, f] - means_cpu[v, f]) / stds_cpu[v, f] : 0.0f0

            # Track for cross-layer correlation
            if layer_sym == :soil
                push!(soil_anomaly_vertices, v)
            elseif layer_sym == :vision
                push!(vision_anomaly_vertices, v)
            end

            push!(results, Dict{String,Any}(
                "vertex_id" => vid,
                "layer" => string(layer_sym),
                "feature" => fname,
                "anomaly_type" => anomaly_type_from_layer(layer_sym),
                "severity" => severity,
                "current_value" => Float64(current_cpu[v, f]),
                "rolling_mean" => Float64(means_cpu[v, f]),
                "rolling_std" => Float64(stds_cpu[v, f]),
                "sigma_deviation" => Float64(sigma_dev),
                "anomaly_rules" => fired_rules,
                "timestamp_start" => string(ts_start),
                "timestamp_end" => string(ts_end),
                "cross_layer_confirmed" => false,  # updated below
            ))
        end

        # Vision layer: additionally flag high anomaly_score even without history deviation
        if layer_sym == :vision && d >= 3
            vision_feat_cpu = ensure_cpu(layer.vertex_features)
            for v in 1:nv
                enabled_vertex[v] || continue
                if vision_feat_cpu[v, 3] > vision_score_vertex[v]  # high CV anomaly score
                    push!(vision_anomaly_vertices, v)
                end
            end
        end

        # --- Contiguous outage detection (NaN-run based) ---
        layer_outage_vertices = Set{Int}()
        for v in 1:nv, f in 1:d
            enabled_vertex[v] || continue

            min_nan_run = min_nan_run_vertex[v]
            nan_runs_cpu[v, f] < min_nan_run && continue

            vid = v <= length(layer.vertex_ids) ? layer.vertex_ids[v] : "v_$v"
            fname = f <= length(fnames) ? fnames[f] : "feature_$f"
            run_len = Int(nan_runs_cpu[v, f])
            severity = run_len >= 2 * min_nan_run ? "alarm" : "warning"

            sigma_dev = stds_cpu[v, f] > 1.0f-8 ?
                        abs(current_cpu[v, f] - means_cpu[v, f]) / stds_cpu[v, f] : 0.0f0

            push!(layer_outage_vertices, v)
            push!(results, Dict{String,Any}(
                "vertex_id" => vid,
                "layer" => string(layer_sym),
                "feature" => fname,
                "anomaly_type" => OUTAGE_ANOMALY_TYPE,
                "severity" => severity,
                "current_value" => Float64(current_cpu[v, f]),
                "rolling_mean" => Float64(means_cpu[v, f]),
                "rolling_std" => Float64(stds_cpu[v, f]),
                "sigma_deviation" => Float64(sigma_dev),
                "anomaly_rules" => ["contiguous_nan_run"],
                "nan_run_length" => run_len,
                "estimated_outage_minutes" => run_len * CADENCE_MINUTES,
                "timestamp_start" => string(ts_start),
                "timestamp_end" => string(ts_end),
                "cross_layer_confirmed" => false,
                "multi_layer_outage" => false,
            ))
        end
        outage_vertices[layer_sym] = layer_outage_vertices
    end

    # Cross-layer correlation: soil + vision anomaly on same vertex → escalate
    cross_confirmed = intersect(soil_anomaly_vertices, vision_anomaly_vertices)
    if !isempty(cross_confirmed)
        for r in results
            vid = r["vertex_id"]
            # Find vertex index
            vidx = get(graph.vertex_index, vid, 0)
            if vidx in cross_confirmed
                r["cross_layer_confirmed"] = true
                # Escalate warnings to alarms
                if r["severity"] == "warning"
                    r["severity"] = "alarm"
                end
            end
        end
    end

    # Multi-layer outage correlation
    if !isempty(outage_vertices)
        all_outage_verts = reduce(∪, values(outage_vertices); init=Set{Int}())
        for v in all_outage_verts
            layers_with_outage = [sym for (sym, s) in outage_vertices if v in s]
            if length(layers_with_outage) >= 2
                for r in results
                    vidx = get(graph.vertex_index, r["vertex_id"], 0)
                    if vidx == v && get(r, "anomaly_type", "") == OUTAGE_ANOMALY_TYPE
                        r["multi_layer_outage"] = true
                    end
                end
            end
        end
    end

    return results
end
