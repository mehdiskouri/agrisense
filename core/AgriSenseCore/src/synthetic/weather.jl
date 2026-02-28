# ---------------------------------------------------------------------------
# Synthetic weather data generator
# ---------------------------------------------------------------------------

"""
    generate_weather_data(n_stations, n_steps; seed) -> Dict

Generate synthetic temperature, humidity, precipitation, and wind time-series.
"""
function generate_weather_data(n_stations::Int, n_steps::Int; seed::Int=42)::Dict{String,Any}
    # Stub — full implementation in Phase 4
    return Dict{String,Any}("layer" => "weather", "n_stations" => n_stations, "n_steps" => n_steps)
end
