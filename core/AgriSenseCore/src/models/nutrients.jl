# ---------------------------------------------------------------------------
# NPK deficit scoring (GPU-portable)
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

Score NPK deficits per zone against crop stage requirements.
"""
function compute_nutrient_report(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    has_npk  = haskey(graph.layers, :npk)
    has_crop = haskey(graph.layers, :crop_requirements)

    (!has_npk || !has_crop) && return Dict{String,Any}[]

    nv = graph.n_vertices
    npk_layer  = graph.layers[:npk]
    crop_layer = graph.layers[:crop_requirements]

    # Current NPK readings — cols 1-3 of :npk layer
    current_n = Float32.(npk_layer.vertex_features[:, 1])
    current_p = Float32.(npk_layer.vertex_features[:, 2])
    current_k = Float32.(npk_layer.vertex_features[:, 3])

    # Required NPK — cols 3-5 of :crop_requirements layer
    req_n = Float32.(crop_layer.vertex_features[:, 3])
    req_p = Float32.(crop_layer.vertex_features[:, 4])
    req_k = Float32.(crop_layer.vertex_features[:, 5])

    # Compute deficits (CPU — graph data is always on CPU)
    deficit_n = similar(current_n)
    deficit_p = similar(current_p)
    deficit_k = similar(current_k)
    @. deficit_n = max(req_n - current_n, 0.0f0)
    @. deficit_p = max(req_p - current_p, 0.0f0)
    @. deficit_k = max(req_k - current_k, 0.0f0)

    # Growth-stage sensitivity weight: early growth → higher sensitivity
    growth_progress = Float32.(crop_layer.vertex_features[:, 2])
    growth_weight = @. 1.5f0 - 0.5f0 * clamp(growth_progress, 0.0f0, 1.0f0)

    # Severity scores per nutrient
    sev_n = deficit_n .* growth_weight
    sev_p = deficit_p .* growth_weight
    sev_k = deficit_k .* growth_weight

    # Overall severity: mean of 3 nutrient severities, normalised to [0, 1]
    # Normalize by max possible deficit (assume required ≤ 1.0 for normalization)
    max_deficit = max(maximum(req_n), maximum(req_p), maximum(req_k), 1.0f0)
    overall_severity = @. (sev_n + sev_p + sev_k) / (3.0f0 * max_deficit * 1.5f0)
    overall_severity .= clamp.(overall_severity, 0.0f0, 1.0f0)

    # Vision confirmation boost
    has_vision = haskey(graph.layers, :vision)
    vision_boosted = fill(false, nv)
    if has_vision
        vision_layer = graph.layers[:vision]
        # col 3 = anomaly_score
        if size(vision_layer.vertex_features, 2) >= 3
            anomaly_scores = Float32.(vision_layer.vertex_features[:, 3])
            for v in 1:nv
                if anomaly_scores[v] > 0.5f0  # threshold
                    overall_severity[v] = clamp(overall_severity[v] * 2.0f0, 0.0f0, 1.0f0)
                    vision_boosted[v] = true
                end
            end
        end
    end

    # Aggregate per zone via :npk layer edges
    results = Dict{String,Any}[]
    ne = size(npk_layer.incidence, 2)
    for e in 1:ne
        members = findall(!iszero, @view npk_layer.incidence[:, e])
        isempty(members) && continue

        zone_id = e <= length(npk_layer.edge_ids) ?
                  npk_layer.edge_ids[e] : "zone_$e"

        zone_n_def = mean(deficit_n[members])
        zone_p_def = mean(deficit_p[members])
        zone_k_def = mean(deficit_k[members])
        zone_sev   = mean(overall_severity[members])
        zone_vis   = any(vision_boosted[members])

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
