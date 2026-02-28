using Test
using AgriSenseCore
using SparseArrays

@testset "Type Constructors" begin

    @testset "ZoneConfig from Dict" begin
        d = Dict{String,Any}(
            "id" => "z1", "name" => "Bay-1",
            "zone_type" => "greenhouse", "area_m2" => 150.0,
            "soil_type" => "loam",
        )
        zc = ZoneConfig(d)
        @test zc.id == "z1"
        @test zc.name == "Bay-1"
        @test zc.zone_type == :greenhouse
        @test zc.area_m2 ≈ 150.0
        @test zc.soil_type == "loam"
    end

    @testset "ZoneConfig defaults soil_type to 'unknown'" begin
        d = Dict{String,Any}("id" => "z2", "name" => "Bay-2",
                              "zone_type" => "open_field", "area_m2" => 50.0)
        zc = ZoneConfig(d)
        @test zc.soil_type == "unknown"
    end

    @testset "ModelConfig from Dict (defaults true)" begin
        mc = ModelConfig(Dict{String,Any}())
        @test mc.irrigation == true
        @test mc.nutrients == true
        @test mc.yield_forecast == true
        @test mc.anomaly_detection == true
    end

    @testset "ModelConfig partial override" begin
        mc = ModelConfig(Dict{String,Any}("irrigation" => false))
        @test mc.irrigation == false
        @test mc.nutrients == true
    end

    @testset "FarmProfile from Dict" begin
        d = Dict{String,Any}(
            "farm_id" => "f1",
            "farm_type" => "hybrid",
            "active_layers" => ["soil", "irrigation", "weather"],
            "zones" => [
                Dict("id" => "z1", "name" => "Z1", "zone_type" => "open_field", "area_m2" => 80.0),
            ],
            "models" => Dict{String,Any}("yield_forecast" => false),
        )
        fp = FarmProfile(d)
        @test fp.farm_id == "f1"
        @test fp.farm_type == :hybrid
        @test :soil in fp.active_layers
        @test :weather in fp.active_layers
        @test length(fp.zones) == 1
        @test fp.models.yield_forecast == false
    end

    @testset "FarmProfile with empty zones/models" begin
        d = Dict{String,Any}(
            "farm_id" => "f2",
            "farm_type" => "greenhouse",
        )
        fp = FarmProfile(d)
        @test isempty(fp.zones)
        @test isempty(fp.active_layers)
        @test fp.models.irrigation == true  # default
    end

    @testset "HyperGraphLayer type parameters" begin
        # Ensure the struct is parametric on matrix types
        layer = HyperGraphLayer(
            sparse(Int32[1, 2], Int32[1, 1], Float32[1.0, 1.0], 3, 1),
            zeros(Float32, 3, 4),
            [Dict{String,Any}()],
            ["v1", "v2", "v3"],
            ["e1"],
        )
        @test layer isa HyperGraphLayer
        @test size(layer.incidence) == (3, 1)
        @test size(layer.vertex_features) == (3, 4)
    end
end
