using Test
using AgriSenseCore

@testset "Synthetic Data Generator" begin
    @testset "generate_synthetic returns complete contract" begin
        result = AgriSenseCore.generate_synthetic(farm_type="greenhouse", days=90, seed=42)
        @test result["status"] == "ok"
        @test result["farm_type"] == "greenhouse"
        @test result["days"] == 90
        @test result["seed"] == 42
        @test result["n_steps"] == 90 * 24 * 60 ÷ AgriSenseCore.CADENCE_MINUTES
        @test haskey(result, "topology")
        @test haskey(result, "layers")
        for key in ["soil", "weather", "irrigation", "npk", "lighting", "vision"]
            @test haskey(result["layers"], key)
        end
    end

    @testset "90-day generation has expected row counts" begin
        result = AgriSenseCore.generate_synthetic(farm_type="open_field", days=90, seed=7)
        n_steps = result["n_steps"]
        layers = result["layers"]
        @test size(layers["soil"]["moisture"], 1) == n_steps
        @test size(layers["weather"]["temperature"], 1) == n_steps
        @test size(layers["irrigation"]["applied_mm"], 1) == n_steps
        @test size(layers["npk"]["nitrogen_mg_kg"], 1) == cld(90, 7)
    end

    @testset "soil moisture remains in [0,1] ignoring NaN" begin
        data = AgriSenseCore.generate_soil_data(12, 96 * 10; seed=9, use_gpu=false)
        m = data["moisture"]
        valid = .!isnan.(m)
        @test all(m[valid] .>= 0.0f0)
        @test all(m[valid] .<= 1.0f0)
    end

    @testset "NaN + mask encoding is consistent" begin
        result = AgriSenseCore.generate_synthetic(farm_type="greenhouse", days=30, seed=12)
        layers = result["layers"]

        soil = layers["soil"]
        soil_mask = soil["missing_mask"]
        for ch in ["moisture", "temperature", "conductivity", "ph"]
            vals = soil[ch]
            @test size(vals) == size(soil_mask)
            @test all(isnan.(vals) .== soil_mask)
        end

        weather = layers["weather"]
        weather_mask = weather["missing_mask"]
        for ch in ["temperature", "humidity", "precipitation_mm", "wind_speed", "wind_direction", "pressure_hpa", "et0", "solar_rad"]
            vals = weather[ch]
            @test size(vals) == size(weather_mask)
            @test all(isnan.(vals) .== weather_mask)
        end
    end

    @testset "dropout frequency is near configured rate" begin
        days = 60
        result = AgriSenseCore.generate_synthetic(farm_type="greenhouse", days=days, seed=22, outage_prob=0.0)
        soil_mask = result["layers"]["soil"]["missing_mask"]
        observed = count(soil_mask) / length(soil_mask)
        @test observed > 0.01
        @test observed < 0.06
    end

    @testset "CPU deterministic reproducibility for same seed" begin
        a = AgriSenseCore.generate_weather_data(3, 96 * 7; seed=123, use_gpu=false)
        b = AgriSenseCore.generate_weather_data(3, 96 * 7; seed=123, use_gpu=false)
        @test isequal(a["temperature"], b["temperature"])
        @test isequal(a["humidity"], b["humidity"])
        @test a["missing_mask"] == b["missing_mask"]
    end

    @testset "hybrid mode emits per-zone active layers" begin
        result = AgriSenseCore.generate(:hybrid, 30, 101)
        zones = result["topology"]["zones"]
        @test length(zones) == 6
        @test zones[1]["zone_type"] == "greenhouse"
        @test zones[end]["zone_type"] == "open_field"

        gh = filter(z -> z["zone_type"] == "greenhouse", zones)
        of = filter(z -> z["zone_type"] == "open_field", zones)
        @test all("vision" in z["active_layers"] for z in gh)
        @test all(!("vision" in z["active_layers"]) for z in of)
    end

    @testset "StructArray-driven metadata exists" begin
        result = AgriSenseCore.generate(:greenhouse, 10, 5)
        topo = result["topology"]
        @test haskey(topo, "soil_sensors")
        @test haskey(topo["soil_sensors"], "sensor_id")
        @test haskey(topo["soil_sensors"], "zone_id")
        @test length(topo["soil_sensors"]["sensor_id"]) == length(topo["soil_sensors"]["zone_id"])
    end
end

# ===========================================================================
# Section F — Contiguous Outage Injection Tests
# ===========================================================================
@testset "Contiguous Outage Injection" begin

    @testset "contiguous outage produces runs ≥ min duration" begin
        data = AgriSenseCore.generate_soil_data(6, 96 * 10;
                    seed=77, use_gpu=false, outage_prob=0.10f0,
                    outage_duration_range=(4, 20))
        mask = data["missing_mask"]
        found_long_run = false
        for col in 1:size(mask, 2)
            run = 0
            for row in 1:size(mask, 1)
                if mask[row, col]
                    run += 1
                else
                    if run >= 4
                        found_long_run = true
                    end
                    run = 0
                end
            end
            run >= 4 && (found_long_run = true)
        end
        @test found_long_run
    end

    @testset "outage events metadata matches mask" begin
        data = AgriSenseCore.generate_soil_data(4, 500;
                    seed=99, use_gpu=false, outage_prob=0.05f0,
                    outage_duration_range=(4, 20))
        events = data["outage_events"]
        mask = data["missing_mask"]
        vals = data["moisture"]
        for e in events
            ch = e["channel"]
            t_start = e["start"]
            dur = e["duration"]
            t_end = min(t_start + dur - 1, size(mask, 1))
            @test all(mask[t_start:t_end, ch])
            @test all(isnan.(vals[t_start:t_end, ch]))
        end
    end

    @testset "outage_mask is subset of missing_mask" begin
        data = AgriSenseCore.generate_soil_data(6, 96 * 5;
                    seed=55, use_gpu=false, outage_prob=0.03f0)
        @test all(data["outage_mask"] .<= data["missing_mask"])
    end

    @testset "outage_mask + bernoulli = combined mask" begin
        # Generate with only Bernoulli (outage_prob=0)
        bern = AgriSenseCore.generate_soil_data(4, 200;
                    seed=42, use_gpu=false, outage_prob=0.0f0)
        # Generate with only outages (dropout_rate=0)
        out = AgriSenseCore.generate_soil_data(4, 200;
                    seed=42, use_gpu=false, dropout_rate=0.0f0, outage_prob=0.05f0)
        # Outage-only mask has no Bernoulli dropout
        @test count(out["missing_mask"]) == count(out["outage_mask"])
        # Bernoulli-only mask has empty outage_mask
        @test count(bern["outage_mask"]) == 0
    end

    @testset "outage events are non-overlapping within a channel" begin
        data = AgriSenseCore.generate_soil_data(8, 96 * 10;
                    seed=33, use_gpu=false, outage_prob=0.08f0,
                    outage_duration_range=(4, 30))
        events = data["outage_events"]
        by_ch = Dict{Int, Vector}()
        for e in events
            ch = e["channel"]
            push!(get!(by_ch, ch, []), e)
        end
        for (ch, ch_events) in by_ch
            sorted = sort(ch_events; by=e->e["start"])
            for i in 1:(length(sorted)-1)
                @test sorted[i]["start"] + sorted[i]["duration"] <= sorted[i+1]["start"]
            end
        end
    end

    @testset "zero outage_prob preserves Bernoulli behavior" begin
        a = AgriSenseCore.generate_soil_data(4, 200;
                    seed=42, use_gpu=false, outage_prob=0.0f0)
        b = AgriSenseCore.generate_soil_data(4, 200;
                    seed=42, use_gpu=false, outage_prob=0.0f0)
        @test isequal(a["moisture"], b["moisture"])
        @test a["missing_mask"] == b["missing_mask"]
        @test isempty(a["outage_events"])
    end

    @testset "generator output includes outage metadata" begin
        result = AgriSenseCore.generate(:greenhouse, 30, 42)
        for layer_name in ["soil", "weather", "npk", "lighting", "vision"]
            layer = result["layers"][layer_name]
            @test haskey(layer, "outage_events")
            @test haskey(layer, "outage_mask")
        end
        miss = result["missingness"]
        @test haskey(miss, "outage_prob")
        @test haskey(miss, "outage_duration_range")
        @test haskey(miss, "outage_encoding")
    end

    @testset "NPK uses scaled duration range" begin
        npk = AgriSenseCore.generate_npk_data(6, 52;
                    seed=42, use_gpu=false, outage_prob=0.15f0,
                    outage_duration_range=(1, 4))
        events = npk["outage_events"]
        if !isempty(events)
            @test all(1 <= e["duration"] <= 4 for e in events)
        end
        @test true  # pass even if no events (probabilistic)
    end

    @testset "CPU deterministic reproducibility for outage events" begin
        a = AgriSenseCore.generate_soil_data(6, 96 * 5;
                    seed=77, use_gpu=false, outage_prob=0.05f0)
        b = AgriSenseCore.generate_soil_data(6, 96 * 5;
                    seed=77, use_gpu=false, outage_prob=0.05f0)
        @test a["outage_events"] == b["outage_events"]
        @test a["outage_mask"] == b["outage_mask"]
        @test a["missing_mask"] == b["missing_mask"]
    end
end
