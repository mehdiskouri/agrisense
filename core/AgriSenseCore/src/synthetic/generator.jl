# ---------------------------------------------------------------------------
# Master synthetic data generator — dispatches per layer
# ---------------------------------------------------------------------------

"""
    generate(farm_type::Symbol, days::Int, seed::Int) -> Dict

Generate a complete synthetic dataset for a demo farm.
"""
function generate(farm_type::Symbol, days::Int, seed::Int)::Dict{String,Any}
    # Stub — full implementation in Phase 4
    return Dict{String,Any}(
        "farm_type" => string(farm_type),
        "days" => days,
        "seed" => seed,
        "status" => "stub",
    )
end
