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
            # Compute mean
            s = 0.0f0
            for t in 1:valid_len
                idx = t <= buf_size ? t : mod1(t, buf_size)
                s += history[v, f, idx]
            end
            m = s / Float32(valid_len)
            means[v, f] = m

            # Compute std
            ss = 0.0f0
            for t in 1:valid_len
                idx = t <= buf_size ? t : mod1(t, buf_size)
                diff = history[v, f, idx] - m
                ss += diff * diff
            end
            stds[v, f] = sqrt(ss / Float32(valid_len))
        end
    end
end

@kernel function western_electric_kernel!(flags, current, means, stds,
                                          history, head_arr, valid_len_arr, buf_size,
                                          sigma1, sigma2, sigma3)
    i = @index(Global)  # flat index over (vertex, feature)
    @inbounds begin
        d = size(current, 2)
        v = div(i - 1, d) + 1
        f = mod(i - 1, d) + 1
        s = stds[v, f]
        m = means[v, f]
        head = Int32(head_arr[1])
        valid_len = Int32(valid_len_arr[1])

        flag = Int32(0)  # bitfield: bit0=3σ, bit1=2of3>2σ, bit2=4of5>1σ, bit3=8consec

        if s >= 1.0f-8
            cur = current[v, f]
            deviation = abs(cur - m)

            # --- Rule 1: single point > 3σ ---
            if deviation > sigma3 * s
                flag = flag | Int32(1)  # bit 0
            end

            # Ring buffer: head points to NEXT write slot, so most recent is head-1

            # --- Rule 2: 2 of last 3 > 2σ ---
            if valid_len >= 2
                count_2sigma = Int32(deviation > sigma2 * s ? 1 : 0)
                for k in 1:min(Int32(2), valid_len)
                    idx = mod1(head - k, buf_size)
                    hval = history[v, f, idx]
                    if abs(hval - m) > sigma2 * s
                        count_2sigma += Int32(1)
                    end
                end
                if count_2sigma >= Int32(2)
                    flag = flag | Int32(2)  # bit 1
                end
            end

            # --- Rule 3: 4 of last 5 > 1σ ---
            if valid_len >= 4
                count_1sigma = Int32(deviation > sigma1 * s ? 1 : 0)
                for k in 1:min(Int32(4), valid_len)
                    idx = mod1(head - k, buf_size)
                    hval = history[v, f, idx]
                    if abs(hval - m) > sigma1 * s
                        count_1sigma += Int32(1)
                    end
                end
                if count_1sigma >= Int32(4)
                    flag = flag | Int32(4)  # bit 2
                end
            end

            # --- Rule 4: 8 consecutive same side of mean ---
            if valid_len >= 7
                cur_side = Int32(cur > m ? 1 : -1)
                all_same = true
                for k in 1:min(Int32(7), valid_len)
                    idx = mod1(head - k, buf_size)
                    hval = history[v, f, idx]
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
        end  # s >= 1.0f-8

        flags[v, f] = flag
    end
end

# ---------------------------------------------------------------------------
# Minimum history required for Western Electric rules
# ---------------------------------------------------------------------------
const MIN_HISTORY_FOR_ANOMALY = 8

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
    compute_anomaly_detection(graph::LayeredHyperGraph) -> Vector{Dict}

Detect anomalies using rolling mean/σ and Western Electric rules on sensor streams.
GPU-accelerated: kernels compute on device, results pulled to CPU for Dict output.
"""
function compute_anomaly_detection(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    results = Dict{String,Any}[]
    nv = graph.n_vertices

    # Track which vertices have anomalies per layer for cross-layer correlation
    soil_anomaly_vertices = Set{Int}()
    vision_anomaly_vertices = Set{Int}()

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

    sigma1 = 1.0f0
    sigma2 = 2.0f0
    sigma3 = 3.0f0

    # Rule name lookup for bitfield decoding
    rule_labels = ("3sigma", "2of3_2sigma", "4of5_1sigma", "8consec_same_side")

    for (layer_sym, layer) in graph.layers
        layer.history_length < MIN_HISTORY_FOR_ANOMALY && continue

        d = size(layer.vertex_features, 2)
        backend = array_backend(layer.vertex_features)
        ndrange = nv * d

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
        head_val = Int32(mod1(layer.history_length, size(layer.feature_history, 3)))
        head_arr = backend isa CPU ? Int32[head_val] :
                   (HAS_CUDA ? CuArray(Int32[head_val]) : Int32[head_val])
        buf_size = Int32(size(layer.feature_history, 3))

        # Rolling stats kernel
        launch_kernel!(rolling_stats_kernel!, backend, ndrange,
                       means, stds, layer.feature_history, valid_len_arr, Int32(d))

        # Western Electric kernel (full 4-rule bitfield)
        launch_kernel!(western_electric_kernel!, backend, ndrange,
                       flags, current, means, stds,
                       layer.feature_history, head_arr, valid_len_arr, buf_size,
                       sigma1, sigma2, sigma3)

        # Pull results to CPU for Dict building
        flags_cpu   = ensure_cpu(flags)
        means_cpu   = ensure_cpu(means)
        stds_cpu    = ensure_cpu(stds)
        current_cpu = ensure_cpu(current)

        # Collect flagged anomalies
        fnames = get(feature_names, layer_sym, ["feature_$i" for i in 1:d])

        # Timestamp range from cadence constant
        ts_end   = now(UTC)
        ts_start = ts_end - Minute(CADENCE_MINUTES * layer.history_length)

        for v in 1:nv, f in 1:d
            flags_cpu[v, f] == 0 && continue
            bitflag = flags_cpu[v, f]

            # Rule 3 (4-of-5 > 1σ) alone is intentionally treated as low-signal noise.
            # Require corroboration from stronger rules to avoid unstable baseline alerts.
            bitflag == Int32(4) && continue

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
                if vision_feat_cpu[v, 3] > 0.7f0  # high CV anomaly score
                    push!(vision_anomaly_vertices, v)
                end
            end
        end
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

    return results
end
