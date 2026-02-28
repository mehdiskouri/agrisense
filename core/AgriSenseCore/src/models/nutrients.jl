# ---------------------------------------------------------------------------
# NPK deficit scoring (GPU-portable)
# ---------------------------------------------------------------------------

using KernelAbstractions
using StaticArrays

@kernel function npk_deficit_kernel!(deficit, current_npk, required_npk)
    i = @index(Global)
    @inbounds begin
        deficit[i] = max(required_npk[i] - current_npk[i], 0.0f0)
    end
end

"""
    compute_nutrient_report(graph::LayeredHyperGraph) -> Vector{Dict}

Score NPK deficits per zone against crop stage requirements.
"""
function compute_nutrient_report(graph::LayeredHyperGraph)::Vector{Dict{String,Any}}
    # Stub — full implementation in Phase 3
    return Dict{String,Any}[]
end
