# ---------------------------------------------------------------------------
# Master synthetic data generator — dispatches per layer
# ---------------------------------------------------------------------------

"""
    generate(farm_type::Symbol, days::Int, seed::Int) -> Dict

Generate a complete synthetic dataset for a demo farm.
"""
function _compute_synthetic_yield_oracle(
    n_steps::Int,
    n_beds::Int,
    n_zones::Int,
    soil::Dict{String,Any},
    weather::Dict{String,Any},
    npk::Dict{String,Any},
    lighting::Dict{String,Any},
    vision::Dict{String,Any},
)::Dict{String,Any}
    bed_zone_index = Int[mod(i - 1, n_zones) + 1 for i in 1:n_beds]
    target_yield = Float32[4.0f0 + 0.6f0 * sin(0.5f0 * i) for i in 1:n_beds]

    soil_m = Matrix{Float32}(soil["moisture"])
    weather_t = Matrix{Float32}(weather["temperature"])
    weather_p = Matrix{Float32}(weather["precipitation_mm"])
    npk_n = Matrix{Float32}(npk["nitrogen_mg_kg"])
    npk_pv = Matrix{Float32}(npk["phosphorus_mg_kg"])
    npk_k = Matrix{Float32}(npk["potassium_mg_kg"])
    dli = Matrix{Float32}(lighting["dli_cumulative"])
    anomaly = Matrix{Int8}(vision["anomaly_code"])

    n_soil = size(soil_m, 2)
    n_weather = size(weather_t, 2)
    n_lights = size(dli, 2)
    n_weeks = max(1, size(npk_n, 1))
    cadence_minutes = Int(get(soil, "cadence_minutes", CADENCE_MINUTES))

    yield_oracle = zeros(Float32, n_steps, n_beds)
    for t in 1:n_steps
        week_idx = clamp(Int(cld(t * cadence_minutes, 7 * 24 * 60)), 1, n_weeks)
        for b in 1:n_beds
            zone_idx = bed_zone_index[b]
            soil_col = mod(zone_idx - 1, max(1, n_soil)) + 1
            weather_col = mod(zone_idx - 1, max(1, n_weather)) + 1
            light_col = n_lights == 0 ? 1 : (mod(b - 1, n_lights) + 1)

            moisture = soil_m[t, soil_col]
            temp = weather_t[t, weather_col]
            precip = weather_p[t, weather_col]
            light_val = n_lights == 0 ? 20.0f0 : dli[t, light_col]
            anomaly_code = size(anomaly, 2) == 0 ? 0 : Int(anomaly[t, mod(b - 1, size(anomaly, 2)) + 1])

            n_val = npk_n[week_idx, zone_idx]
            p_val = npk_pv[week_idx, zone_idx]
            k_val = npk_k[week_idx, zone_idx]

            moisture = isnan(moisture) ? 0.25f0 : moisture
            temp = isnan(temp) ? 22.0f0 : temp
            precip = isnan(precip) ? 0.0f0 : precip
            light_val = isnan(light_val) ? 20.0f0 : light_val
            n_val = isnan(n_val) ? 80.0f0 : n_val
            p_val = isnan(p_val) ? 60.0f0 : p_val
            k_val = isnan(k_val) ? 70.0f0 : k_val

            ks = clamp((moisture - 0.15f0) / (0.45f0 - 0.15f0), 0.0f0, 1.0f0)
            kw = temp < 5.0f0 ? 0.0f0 : (temp < 15.0f0 ? (temp - 5.0f0) / 10.0f0 :
                 (temp <= 30.0f0 ? 1.0f0 : (temp < 40.0f0 ? (40.0f0 - temp) / 10.0f0 : 0.0f0)))
            kw = clamp(kw * (1.0f0 + 0.05f0 * clamp(precip / 5.0f0, 0.0f0, 1.0f0)), 0.0f0, 1.0f0)
            kn = clamp(1.0f0 - (max(80.0f0 - n_val, 0.0f0) / 80.0f0 +
                                max(60.0f0 - p_val, 0.0f0) / 60.0f0 +
                                max(70.0f0 - k_val, 0.0f0) / 70.0f0) / 3.0f0, 0.0f0, 1.0f0)
            kl = clamp(light_val / 20.0f0, 0.0f0, 1.0f0)

            stage = clamp(Float32(t) / Float32(max(n_steps, 1)), 0.0f0, 1.0f0)
            growth_scale = clamp(0.35f0 + 0.65f0 * stage, 0.2f0, 1.0f0)
            seasonal = 1.0f0 + 0.08f0 * sin(Float32(2.0f0 * pi) * (Float32(t) / 96.0f0) / 30.0f0)
            anomaly_penalty = anomaly_code <= 0 ? 1.0f0 : (anomaly_code == 1 ? 0.88f0 : 0.82f0)

            quality = 0.35f0 * ks + 0.25f0 * kn + 0.20f0 * kw + 0.20f0 * kl
            yield_oracle[t, b] = max(0.0f0, target_yield[b] * growth_scale * seasonal * quality * anomaly_penalty)
        end
    end

    return Dict{String,Any}(
        "layer" => "yield_oracle",
        "n_steps" => n_steps,
        "n_beds" => n_beds,
        "bed_zone_index" => bed_zone_index,
        "target_yield_kg_m2" => target_yield,
        "yield_kg_m2" => yield_oracle,
    )
end

function generate(farm_type::Symbol, days::Int, seed::Int;
                  outage_prob::Float32=DEFAULT_OUTAGE_PROB,
                  outage_duration_range::Tuple{Int,Int}=DEFAULT_OUTAGE_DURATION_RANGE,
                  )::Dict{String,Any}
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
                                    cadence_minutes=CADENCE_MINUTES,
                                    outage_prob=outage_prob,
                                    outage_duration_range=outage_duration_range)
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
                              cadence_minutes=CADENCE_MINUTES,
                              outage_prob=outage_prob,
                              outage_duration_range=outage_duration_range)

    npk = generate_npk_data(n_zones, n_weeks;
                            seed=seed + 30,
                            use_gpu=use_gpu,
                            dropout_rate=0.01f0,
                            outage_prob=outage_prob,
                            outage_duration_range=(1, 4))

    lighting = n_lighting > 0 ?
        generate_lighting_data(n_lighting, n_steps;
                               seed=seed + 40,
                               use_gpu=use_gpu,
                               dropout_rate=SYNTHETIC_DROPOUT_RATE,
                               cadence_minutes=CADENCE_MINUTES,
                               outage_prob=outage_prob,
                               outage_duration_range=outage_duration_range) :
        Dict{String,Any}("layer" => "lighting", "n_sensors" => 0, "n_steps" => n_steps,
                         "par_umol" => zeros(Float32, n_steps, 0),
                         "dli_cumulative" => zeros(Float32, n_steps, 0),
                         "duty_cycle_pct" => zeros(Float32, n_steps, 0),
                         "spectrum_index" => zeros(Float32, n_steps, 0),
                         "missing_mask" => falses(n_steps, 0),
                         "outage_events" => Dict{String,Any}[],
                         "outage_mask" => falses(n_steps, 0),
                         "cadence_minutes" => CADENCE_MINUTES)

    vision = n_lighting > 0 ?
        generate_vision_data(n_cameras, n_beds, n_steps;
                             seed=seed + 50,
                             use_gpu=use_gpu,
                             dropout_rate=SYNTHETIC_DROPOUT_RATE,
                             outage_prob=outage_prob,
                             outage_duration_range=outage_duration_range) :
        Dict{String,Any}("layer" => "vision", "n_cameras" => 0, "n_beds" => 0, "n_steps" => n_steps,
                         "camera_for_bed" => Int32[],
                         "anomaly_code" => zeros(Int8, n_steps, 0),
                         "anomaly_legend" => Dict("-1" => "missing", "0" => "none", "1" => "pest", "2" => "disease"),
                         "confidence" => zeros(Float32, n_steps, 0),
                         "canopy_coverage_pct" => zeros(Float32, n_steps, 0),
                         "missing_mask" => falses(n_steps, 0),
                         "outage_events" => Dict{String,Any}[],
                         "outage_mask" => falses(n_steps, 0))

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

    yield_oracle = _compute_synthetic_yield_oracle(
        n_steps,
        n_beds,
        n_zones,
        soil,
        weather,
        npk,
        lighting,
        vision,
    )

    return Dict{String,Any}(
        "farm_type" => string(farm_type),
        "days" => days,
        "seed" => seed,
        "cadence_minutes" => CADENCE_MINUTES,
        "n_steps" => n_steps,
        "time_hours" => Vector{Float32}(t_hours),
        "missingness" => Dict("encoding" => "nan_plus_bitmask",
                                "dropout_rate" => SYNTHETIC_DROPOUT_RATE,
                                "outage_prob" => Float64(outage_prob),
                                "outage_duration_range" => [outage_duration_range...],
                                "outage_encoding" => "contiguous_nan_blocks_in_outage_mask"),
        "reproducibility" => Dict("cpu" => "bitwise_deterministic", "gpu" => "statistically_bounded"),
        "topology" => topology,
        "layers" => Dict{String,Any}(
            "soil" => soil,
            "weather" => weather,
            "irrigation" => irrigation,
            "npk" => npk,
            "lighting" => lighting,
            "vision" => vision,
            "yield_oracle" => yield_oracle,
        ),
        "status" => "ok",
    )
end
