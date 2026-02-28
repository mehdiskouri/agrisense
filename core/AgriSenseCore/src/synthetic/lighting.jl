# ---------------------------------------------------------------------------
# Synthetic lighting / PAR data generator
# ---------------------------------------------------------------------------

"""
    generate_lighting_data(n_sensors, n_steps; seed) -> Dict

Generate synthetic PAR, DLI, and spectrum index time-series.
Values follow a diurnal sinusoidal pattern with Gaussian noise.
"""
function generate_lighting_data(n_sensors::Int, n_steps::Int; seed::Int=42)::Dict{String,Any}
    # Stub — full implementation in Phase 4
    return Dict{String,Any}("layer" => "lighting", "n_sensors" => n_sensors, "n_steps" => n_steps)
end
