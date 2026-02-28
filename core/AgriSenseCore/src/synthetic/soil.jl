# ---------------------------------------------------------------------------
# Synthetic soil data generator
# ---------------------------------------------------------------------------

"""
    generate_soil_data(n_sensors, n_steps; seed) -> Dict

Generate synthetic soil moisture, temperature, conductivity, and pH time-series.
"""
function generate_soil_data(n_sensors::Int, n_steps::Int; seed::Int=42)::Dict{String,Any}
    # Stub — full implementation in Phase 4
    return Dict{String,Any}("layer" => "soil", "n_sensors" => n_sensors, "n_steps" => n_steps)
end
