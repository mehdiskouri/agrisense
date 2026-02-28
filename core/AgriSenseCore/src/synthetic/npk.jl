# ---------------------------------------------------------------------------
# Synthetic NPK data generator
# ---------------------------------------------------------------------------

"""
    generate_npk_data(n_zones, n_weeks; seed) -> Dict

Generate synthetic nitrogen, phosphorus, potassium time-series with slow drift
and periodic fertilization step-changes.
"""
function generate_npk_data(n_zones::Int, n_weeks::Int; seed::Int=42)::Dict{String,Any}
    # Stub — full implementation in Phase 4
    return Dict{String,Any}("layer" => "npk", "n_zones" => n_zones, "n_weeks" => n_weeks)
end
