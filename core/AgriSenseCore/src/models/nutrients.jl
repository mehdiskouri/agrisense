# ---------------------------------------------------------------------------
# NPK deficit scoring (GPU-first)
# ---------------------------------------------------------------------------

@kernel function npk_deficit_severity_kernel!(deficit_n, deficit_p, deficit_k,
                                              overall_severity,
                                              current_n, current_p, current_k,
                                              req_n, req_p, req_k,
                                              growth_progress,
                                              w_n::Float32, w_p::Float32, w_k::Float32,
                                              normaliser::Float32)
    i = @index(Global)
    @inbounds begin
        dn = max(req_n[i] - current_n[i], 0.0f0)
        dp = max(req_p[i] - current_p[i], 0.0f0)
        dk = max(req_k[i] - current_k[i], 0.0f0)

        gw = 1.5f0 - 0.5f0 * clamp(growth_progress[i], 0.0f0, 1.0f0)

        deficit_n[i] = dn
        deficit_p[i] = dp
        deficit_k[i] = dk

        sev = (w_n * (dn * gw) + w_p * (dp * gw) + w_k * (dk * gw)) / normaliser
        overall_severity[i] = clamp(sev, 0.0f0, 1.0f0)
    end
end

@kernel function vision_boost_kernel!(overall_severity, vision_boosted,
                                      anomaly_scores, threshold::Float32,
                                      multiplier::Float32)
    i = @index(Global)
    @inbounds begin
        boosted = anomaly_scores[i] > threshold
        vision_boosted[i] = boosted
        if boosted
            overall_severity[i] = clamp(overall_severity[i] * multiplier, 0.0f0, 1.0f0)
        end
    end
end

# ---------------------------------------------------------------------------
# Urgency thresholds
# ---------------------------------------------------------------------------

function severity_to_urgency(s::Float32)::Symbol
    s < 0.25f0 && return :low
    s < 0.50f0 && return :medium
    s < 0.75f0 && return :high
    return :critical
end

function suggest_amendment(n_def::Float32, p_def::Float32, k_def::Float32)::String
    parts = String[]
    n_def > 0 && push!(parts, "nitrogen")
    p_def > 0 && push!(parts, "phosphorus")
    k_def > 0 && push!(parts, "potassium")
    isempty(parts) && return "none"
    return "apply " * join(parts, " + ") * " fertilizer"
end

"""
    compute_nutrient_report(graph; weights=(0.50f0, 0.25f0, 0.25f0)) -> Vector{Dict}

Score NPK deficits per zone against crop stage requirements. GPU-accelerated.

`weights` is a 3-tuple `(w_N, w_P, w_K)` controlling severity emphasis.
Default is N-heavier per agronomic convention (nitrogen is the primary
growth-limiting macronutrient). Weights should sum to 1.
"""
function compute_nutrient_report(graph::LayeredHyperGraph;
                                  weights::NTuple{3,Float32}=(0.50f0, 0.25f0, 0.25f0),
                                  )::Vector{Dict{String,Any}}
    has_npk  = haskey(graph.layers, :npk)
    has_crop = haskey(graph.layers, :crop_requirements)
    (!has_npk || !has_crop) && return Dict{String,Any}[]

    w_n, w_p, w_k = weights

    nv = graph.n_vertices
    npk_layer  = graph.layers[:npk]
    crop_layer = graph.layers[:crop_requirements]
    backend = array_backend(npk_layer.vertex_features)

    # Current NPK readings — cols 1-3 of :npk layer (on device)
    current_n = npk_layer.vertex_features[:, 1]
    current_p = npk_layer.vertex_features[:, 2]
    current_k = npk_layer.vertex_features[:, 3]

    # Required NPK — cols 3-5 of :crop_requirements layer (on device)
    req_n = crop_layer.vertex_features[:, 3]
    req_p = crop_layer.vertex_features[:, 4]
    req_k = crop_layer.vertex_features[:, 5]

    growth_progress = crop_layer.vertex_features[:, 2]

    # Compute deficits + weighted severity via explicit KA kernel
    deficit_n = backend isa CPU ? zeros(Float32, nv) : CUDA.zeros(Float32, nv)
    deficit_p = backend isa CPU ? zeros(Float32, nv) : CUDA.zeros(Float32, nv)
    deficit_k = backend isa CPU ? zeros(Float32, nv) : CUDA.zeros(Float32, nv)

    # N-heavier default: w_n=0.50, w_p=0.25, w_k=0.25 — nitrogen is primary
    # growth-limiting macronutrient.
    max_def_cpu = max(maximum(ensure_cpu(req_n)), maximum(ensure_cpu(req_p)),
                      maximum(ensure_cpu(req_k)), 1.0f0)
    normaliser = Float32(max_def_cpu * 1.5f0)
    overall_severity = backend isa CPU ? zeros(Float32, nv) : CUDA.zeros(Float32, nv)

    launch_kernel!(npk_deficit_severity_kernel!, backend, nv,
                   deficit_n, deficit_p, deficit_k, overall_severity,
                   current_n, current_p, current_k,
                   req_n, req_p, req_k,
                   growth_progress,
                   Float32(w_n), Float32(w_p), Float32(w_k),
                   normaliser)

    # Vision confirmation boost — GPU broadcast
    has_vision = haskey(graph.layers, :vision)
    vision_boosted = if backend isa CPU
        fill(false, nv)
    else
        CUDA.fill(false, nv)
    end
    if has_vision
        vision_layer = graph.layers[:vision]
        if size(vision_layer.vertex_features, 2) >= 3
            anomaly_scores = vision_layer.vertex_features[:, 3]
            launch_kernel!(vision_boost_kernel!, backend, nv,
                           overall_severity, vision_boosted,
                           anomaly_scores, 0.5f0, 2.0f0)
        end
    end

    # Pull everything to CPU for Dict building
    dn_cpu = ensure_cpu(deficit_n)
    dp_cpu = ensure_cpu(deficit_p)
    dk_cpu = ensure_cpu(deficit_k)
    sev_cpu = ensure_cpu(overall_severity)
    vis_cpu = Bool.(ensure_cpu(vision_boosted))

    # Aggregate per zone via :npk layer edges
    results = Dict{String,Any}[]
    B_cpu = ensure_cpu(npk_layer.incidence)
    ne = size(B_cpu, 2)
    for e in 1:ne
        members = findall(!iszero, @view B_cpu[:, e])
        isempty(members) && continue

        zone_id = e <= length(npk_layer.edge_ids) ?
                  npk_layer.edge_ids[e] : "zone_$e"

        zone_n_def = mean(dn_cpu[members])
        zone_p_def = mean(dp_cpu[members])
        zone_k_def = mean(dk_cpu[members])
        zone_sev   = mean(sev_cpu[members])
        zone_vis   = any(vis_cpu[members])

        push!(results, Dict{String,Any}(
            "zone_id" => zone_id,
            "nitrogen_deficit" => Float64(zone_n_def),
            "phosphorus_deficit" => Float64(zone_p_def),
            "potassium_deficit" => Float64(zone_k_def),
            "severity_score" => Float64(zone_sev),
            "urgency" => string(severity_to_urgency(Float32(zone_sev))),
            "suggested_amendment" => suggest_amendment(Float32(zone_n_def),
                                                       Float32(zone_p_def),
                                                       Float32(zone_k_def)),
            "visual_confirmed" => zone_vis,
        ))
    end

    return results
end
