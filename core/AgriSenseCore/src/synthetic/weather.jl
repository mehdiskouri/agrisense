# ---------------------------------------------------------------------------
# Synthetic weather data generator
# ---------------------------------------------------------------------------

"""
    generate_weather_data(n_stations, n_steps; ...) -> Dict

Generates channels `(n_steps, n_stations)`:
temperature, humidity, precipitation_mm, wind_speed, wind_direction,
pressure_hpa, et0, solar_rad with NaN + mask missingness representation.
"""
function generate_weather_data(n_stations::Int, n_steps::Int;
                                seed::Int=42,
                                use_gpu::Bool=HAS_CUDA,
                                dropout_rate::Float32=SYNTHETIC_DROPOUT_RATE,
                                cadence_minutes::Int=CADENCE_MINUTES,
                                )::Dict{String,Any}
    n_stations <= 0 && error("generate_weather_data: n_stations must be > 0")
    n_steps <= 0 && error("generate_weather_data: n_steps must be > 0")

    t = time_grid(n_steps, cadence_minutes)
    th = to_backend(reshape(t, n_steps, 1); use_gpu=use_gpu)
    ω = Float32(2π / 24)

    rng = MersenneTwister(seed + 200)
    station_offset = to_backend(reshape(rand(rng, Float32, n_stations) .* 2.5f0 .- 1.25f0, 1, n_stations); use_gpu=use_gpu)

    noise = correlated_noise(n_steps, n_stations; seed=seed + 201, use_gpu=use_gpu)
    temperature = 19.0f0 .+ station_offset .+ 8.0f0 .* sin.(ω .* th .- 1.0f0) .+
                  1.2f0 .* noise

    # Seasonal rain probability (slow 30-day oscillation)
    season = 0.10f0 .+ 0.08f0 .* sin.(Float32(2π / (30 * 24)) .* th)
    season = clamp.(season, 0.02f0, 0.30f0)
    rrng = MersenneTwister(seed + 202)
    rain_u = to_backend(rand(rrng, Float32, n_steps, n_stations); use_gpu=use_gpu)
    rain_events = Float32.(rain_u .< season)
    rain_amt = to_backend(rand(rrng, Float32, n_steps, n_stations) .* 4.0f0; use_gpu=use_gpu)
    precipitation = rain_events .* rain_amt

    # Humidity anti-correlated with temperature + correlated noise
    humidity = 65.0f0 .- 0.9f0 .* (temperature .- 20.0f0) .+
               4.0f0 .* correlated_noise(n_steps, n_stations; seed=seed + 203, use_gpu=use_gpu)
    humidity = clamp.(humidity, 15.0f0, 100.0f0)

    wind_speed = 2.5f0 .+ 1.0f0 .* abs.(sin.(ω .* th .+ 0.7f0)) .+
                 0.4f0 .* abs.(correlated_noise(n_steps, n_stations; seed=seed + 204, use_gpu=use_gpu))
    wind_speed = clamp.(wind_speed, 0.0f0, 22.0f0)

    wind_direction = mod.(180.0f0 .+ 90.0f0 .* sin.(ω .* th .+ 0.3f0) .+
                          45.0f0 .* correlated_noise(n_steps, n_stations; seed=seed + 205, use_gpu=use_gpu), 360.0f0)

    pressure = 1013.0f0 .+ 6.0f0 .* sin.(ω .* th ./ 2.0f0) .+
               1.8f0 .* correlated_noise(n_steps, n_stations; seed=seed + 206, use_gpu=use_gpu)

    solar_rad = max.(0.0f0, 16.0f0 .* sin.(ω .* th .- 1.2f0))

    # Simple ET0 proxy (aligned with existing irrigation model scale)
    et0 = max.(0.0f0,
               0.0023f0 .* (temperature .+ 17.8f0) .* sqrt.(max.(temperature, 0.0f0)) .* solar_rad)

    temperature_nan, mask = apply_dropout_with_mask(Float32.(temperature);
                                                    rate=dropout_rate,
                                                    seed=seed + 207,
                                                    use_gpu=use_gpu)
    humidity_nan = ifelse.(mask, Float32(NaN), Float32.(humidity))
    precipitation_nan = ifelse.(mask, Float32(NaN), Float32.(precipitation))
    wind_speed_nan = ifelse.(mask, Float32(NaN), Float32.(wind_speed))
    wind_direction_nan = ifelse.(mask, Float32(NaN), Float32.(wind_direction))
    pressure_nan = ifelse.(mask, Float32(NaN), Float32.(pressure))
    et0_nan = ifelse.(mask, Float32(NaN), Float32.(et0))
    solar_rad_nan = ifelse.(mask, Float32(NaN), Float32.(solar_rad))

    return Dict{String,Any}(
        "layer" => "weather",
        "n_stations" => n_stations,
        "n_steps" => n_steps,
        "cadence_minutes" => cadence_minutes,
        "temperature" => cpu_plain(temperature_nan),
        "humidity" => cpu_plain(humidity_nan),
        "precipitation_mm" => cpu_plain(precipitation_nan),
        "wind_speed" => cpu_plain(wind_speed_nan),
        "wind_direction" => cpu_plain(wind_direction_nan),
        "pressure_hpa" => cpu_plain(pressure_nan),
        "et0" => cpu_plain(et0_nan),
        "solar_rad" => cpu_plain(solar_rad_nan),
        "missing_mask" => BitMatrix(cpu_plain(mask)),
    )
end
