# ---------------------------------------------------------------------------
# NPK deficit scoring (GPU-first)
# ---------------------------------------------------------------------------

@kernel function npk_deficit_kernel!(deficit, current_npk, required_npk)
    i = @index(Global)
    @inbounds begin
        deficit[i] = max(required_npk[i] - current_npk[i], 0.0f0)
    end
end

@kernel function severity_score_kernel!(scores, deficit, growth_weight)
    i = @index(Global)
    @inbounds begin
        scores[i] = deficit[i] * growth_weight[i]
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
    compute_nutrient_report(graph::LayeredHyperGraph) -> Vector{Dict}

Score NPK deficits per zone against crop stage requirements. GPU-accelerated.
"""
function compute_nutrient_report(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    has_npk  = haskey(graph.layers, :npk)
    has_crop = haskey(graph.layers, :crop_requirements)
    (!has_npk || !has_crop) && return Dict{String,Any}[]

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

    # Compute deficits via GPU broadcast
    deficit_n = @. max(req_n - current_n, 0.0f0)
    deficit_p = @. max(req_p - current_p, 0.0f0)
    deficit_k = @. max(req_k - current_k, 0.0f0)

    # Growth-stage sensitivity weight
    growth_progress = crop_layer.vertex_features[:, 2]
    growth_weight = @. Float32(1.5f0 - 0.5f0 * clamp(growth_progress, 0.0f0, 1.0f0))

    # Severity scores per nutrient (GPU broadcast)
    sev_n = deficit_n .* growth_weight
    sev_p = deficit_p .* growth_weight
    sev_k = deficit_k .* growth_weight

    # Overall severity (GPU broadcast)
    max_def_cpu = max(maximum(ensure_cpu(req_n)), maximum(ensure_cpu(req_p)),
                      maximum(ensure_cpu(req_k)), 1.0f0)
    overall_severity = @. (sev_n + sev_p + sev_k) / (3.0f0 * max_def_cpu * 1.5f0)
    overall_severity = @. clamp(overall_severity, 0.0f0, 1.0f0)

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
            # GPU-safe broadcast: boost severity where anomaly_score > 0.5
            vision_mask = anomaly_scores .> 0.5f0
            boosted = @. clamp(overall_severity * 2.0f0, 0.0f0, 1.0f0)
            overall_severity = ifelse.(vision_mask, boosted, overall_severity)
            vision_boosted = vision_mask
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
