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
        result = AgriSenseCore.generate_synthetic(farm_type="greenhouse", days=days, seed=22)
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
