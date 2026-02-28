# ---------------------------------------------------------------------------
# Master synthetic data generator — dispatches per layer
# ---------------------------------------------------------------------------

"""
    generate(farm_type::Symbol, days::Int, seed::Int) -> Dict

Generate a complete synthetic dataset for a demo farm.
"""
function generate(farm_type::Symbol, days::Int, seed::Int)::Dict{String,Any}
    days <= 0 && error("generate: days must be > 0")
    farm_type in (:open_field, :greenhouse, :hybrid) ||
        error("generate: farm_type must be :open_field, :greenhouse, or :hybrid")

    n_steps = synthetic_steps(days; cadence_minutes=CADENCE_MINUTES)
    t_hours = time_grid(n_steps, CADENCE_MINUTES)

    # Demo topology sizes (aligned to PRD demonstration farm scale)
    n_zones = farm_type == :greenhouse ? 2 : (farm_type == :open_field ? 4 : 6)
    zone_types = if farm_type == :hybrid
        vcat(fill("greenhouse", 2), fill("open_field", 4))
    else
        fill(string(farm_type), n_zones)
    end

    zones = [Dict{String,Any}(
        "zone_id" => "zone_$(i)",
        "zone_type" => zone_types[i],
        "active_layers" => zone_types[i] == "greenhouse" ?
            ["soil", "irrigation", "lighting", "weather", "crop_requirements", "npk", "vision"] :
            ["soil", "irrigation", "solar", "weather", "crop_requirements", "npk"],
    ) for i in 1:n_zones]

    n_soil_sensors = n_zones * 3
    n_weather_stations = max(1, min(3, n_zones))
    n_lighting = count(==("greenhouse"), zone_types) * 2
    n_cameras = max(1, count(==("greenhouse"), zone_types) * 2)
    n_beds = max(2, n_zones * 2)
    n_weeks = max(1, Int(cld(days, 7)))

    use_gpu = HAS_CUDA

    # Shared global weather forcing first
    weather = generate_weather_data(n_weather_stations, n_steps;
                                    seed=seed + 10,
                                    use_gpu=use_gpu,
                                    dropout_rate=SYNTHETIC_DROPOUT_RATE,
                                    cadence_minutes=CADENCE_MINUTES)
    rain_global = vec(Float32.(mean(weather["precipitation_mm"]; dims=2)))
    rain_global = Float32.(ifelse.(isnan.(rain_global), 0.0f0, rain_global))

    # Synthetic irrigation forcing used by soil model
    rng = MersenneTwister(seed + 11)
    irrigation_global = Float32.(rand(rng, Float32, n_steps) .< 0.045f0) .* (rand(rng, Float32, n_steps) .* 9.0f0 .+ 3.0f0)

    soil = generate_soil_data(n_soil_sensors, n_steps;
                              seed=seed + 20,
                              use_gpu=use_gpu,
                              rainfall_mm=rain_global,
                              irrigation_mm=irrigation_global,
                              dropout_rate=SYNTHETIC_DROPOUT_RATE,
                              cadence_minutes=CADENCE_MINUTES)

    npk = generate_npk_data(n_zones, n_weeks;
                            seed=seed + 30,
                            use_gpu=use_gpu,
                            dropout_rate=0.01f0)

    lighting = n_lighting > 0 ?
        generate_lighting_data(n_lighting, n_steps;
                               seed=seed + 40,
                               use_gpu=use_gpu,
                               dropout_rate=SYNTHETIC_DROPOUT_RATE,
                               cadence_minutes=CADENCE_MINUTES) :
        Dict{String,Any}("layer" => "lighting", "n_sensors" => 0, "n_steps" => n_steps,
                         "par_umol" => zeros(Float32, n_steps, 0),
                         "dli_cumulative" => zeros(Float32, n_steps, 0),
                         "duty_cycle_pct" => zeros(Float32, n_steps, 0),
                         "spectrum_index" => zeros(Float32, n_steps, 0),
                         "missing_mask" => falses(n_steps, 0),
                         "cadence_minutes" => CADENCE_MINUTES)

    vision = n_lighting > 0 ?
        generate_vision_data(n_cameras, n_beds, n_steps;
                             seed=seed + 50,
                             use_gpu=false,
                             dropout_rate=SYNTHETIC_DROPOUT_RATE) :
        Dict{String,Any}("layer" => "vision", "n_cameras" => 0, "n_beds" => 0, "n_steps" => n_steps,
                         "camera_for_bed" => Int32[],
                         "anomaly_code" => zeros(Int8, n_steps, 0),
                         "anomaly_legend" => Dict("-1" => "missing", "0" => "none", "1" => "pest", "2" => "disease"),
                         "confidence" => zeros(Float32, n_steps, 0),
                         "canopy_coverage_pct" => zeros(Float32, n_steps, 0),
                         "missing_mask" => falses(n_steps, 0))

    # StructArrays SoA batches for sensor metadata layout
    soil_sensor_batch = StructArray((
        sensor_id = ["soil_sensor_$(i)" for i in 1:n_soil_sensors],
        zone_id = ["zone_$(mod(i - 1, n_zones) + 1)" for i in 1:n_soil_sensors],
    ))
    station_batch = StructArray((
        station_id = ["weather_station_$(i)" for i in 1:n_weather_stations],
    ))

    irrigation = Dict{String,Any}(
        "layer" => "irrigation",
        "n_valves" => n_zones,
        "n_steps" => n_steps,
        "applied_mm" => reshape(irrigation_global, n_steps, 1) .* ones(Float32, 1, n_zones),
    )

    topology = Dict{String,Any}(
        "n_zones" => n_zones,
        "zones" => zones,
        "soil_sensors" => Dict(
            "sensor_id" => collect(soil_sensor_batch.sensor_id),
            "zone_id" => collect(soil_sensor_batch.zone_id),
        ),
        "weather_stations" => Dict(
            "station_id" => collect(station_batch.station_id),
        ),
    )

    return Dict{String,Any}(
        "farm_type" => string(farm_type),
        "days" => days,
        "seed" => seed,
        "cadence_minutes" => CADENCE_MINUTES,
        "n_steps" => n_steps,
        "time_hours" => Vector{Float32}(t_hours),
        "missingness" => Dict("encoding" => "nan_plus_bitmask", "dropout_rate" => SYNTHETIC_DROPOUT_RATE),
        "reproducibility" => Dict("cpu" => "bitwise_deterministic", "gpu" => "statistically_bounded"),
        "topology" => topology,
        "layers" => Dict{String,Any}(
            "soil" => soil,
            "weather" => weather,
            "irrigation" => irrigation,
            "npk" => npk,
            "lighting" => lighting,
            "vision" => vision,
        ),
        "status" => "ok",
    )
end
