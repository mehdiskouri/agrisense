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
            zeros(Float32, 3, 4, DEFAULT_HISTORY_SIZE),
            1, 0,
            [Dict{String,Any}()],
            ["v1", "v2", "v3"],
            ["e1"],
        )
        @test layer isa HyperGraphLayer
        @test size(layer.incidence) == (3, 1)
        @test size(layer.vertex_features) == (3, 4)
        @test size(layer.feature_history) == (3, 4, DEFAULT_HISTORY_SIZE)
        @test layer.history_head == 1
        @test layer.history_length == 0
    end
end

# ===========================================================================
# Phase 13 — Mask Infrastructure Tests
# ===========================================================================
@testset "Phase 13 — Mask Fields" begin

    @testset "8-arg constructor creates false-filled masks" begin
        layer = HyperGraphLayer(
            sparse(Int32[1], Int32[1], Float32[1.0], 2, 1),
            zeros(Float32, 2, 3),
            zeros(Float32, 2, 3, 10),
            1, 0,
            [Dict{String,Any}()], ["v1","v2"], ["e1"],
        )
        @test size(layer.feature_history_mask) == (2, 3, 10)
        @test size(layer.vertex_features_mask) == (2, 3)
        @test all(layer.feature_history_mask .== false)
        @test all(layer.vertex_features_mask .== false)
    end

    @testset "push_features! sets mask to true for valid data" begin
        layer = HyperGraphLayer(
            sparse(Int32[1], Int32[1], Float32[1.0], 2, 1),
            zeros(Float32, 2, 3),
            zeros(Float32, 2, 3, 5),
            1, 0,
            [Dict{String,Any}()], ["v1","v2"], ["e1"],
        )
        push_features!(layer, 1, Float32[1.0, 2.0, 3.0])
        # All valid → mask should be true
        @test all(layer.vertex_features_mask[1, :])
        @test all(layer.feature_history_mask[1, :, 1])
        # Vertex 2 untouched → still false
        @test all(layer.vertex_features_mask[2, :] .== false)
    end

    @testset "push_features! sets mask to false for NaN data" begin
        layer = HyperGraphLayer(
            sparse(Int32[1], Int32[1], Float32[1.0], 1, 1),
            zeros(Float32, 1, 3),
            zeros(Float32, 1, 3, 5),
            1, 0,
            [Dict{String,Any}()], ["v1"], ["e1"],
        )
        push_features!(layer, 1, Float32[NaN, 2.0, NaN])
        @test layer.vertex_features_mask[1, 1] == false  # NaN
        @test layer.vertex_features_mask[1, 2] == true   # valid
        @test layer.vertex_features_mask[1, 3] == false  # NaN
        @test layer.feature_history_mask[1, 1, 1] == false
        @test layer.feature_history_mask[1, 2, 1] == true
        @test layer.feature_history_mask[1, 3, 1] == false
    end

    @testset "mask tracks mixed valid/NaN pushes across time" begin
        layer = HyperGraphLayer(
            sparse(Int32[1], Int32[1], Float32[1.0], 1, 1),
            zeros(Float32, 1, 2),
            zeros(Float32, 1, 2, 4),
            1, 0,
            [Dict{String,Any}()], ["v1"], ["e1"],
        )
        # Push 1: both valid
        push_features!(layer, 1, Float32[1.0, 2.0])
        @test layer.feature_history_mask[1, 1, 1] == true
        @test layer.feature_history_mask[1, 2, 1] == true

        # Push 2: first is NaN
        push_features!(layer, 1, Float32[NaN, 3.0])
        @test layer.feature_history_mask[1, 1, 2] == false
        @test layer.feature_history_mask[1, 2, 2] == true

        # Snapshot mask reflects last push
        @test layer.vertex_features_mask[1, 1] == false
        @test layer.vertex_features_mask[1, 2] == true
    end

    @testset "mask wraps correctly with ring buffer" begin
        buf_size = 3
        layer = HyperGraphLayer(
            sparse(Int32[1], Int32[1], Float32[1.0], 1, 1),
            zeros(Float32, 1, 1),
            zeros(Float32, 1, 1, buf_size),
            1, 0,
            [Dict{String,Any}()], ["v1"], ["e1"],
        )
        # Push: valid, NaN, valid, valid (wraps at 4th)
        push_features!(layer, 1, Float32[1.0])  # slot 1 → true
        push_features!(layer, 1, Float32[NaN])   # slot 2 → false
        push_features!(layer, 1, Float32[3.0])   # slot 3 → true
        push_features!(layer, 1, Float32[4.0])   # slot 1 (wrap) → true

        @test layer.feature_history_mask[1, 1, 1] == true   # overwritten by push 4
        @test layer.feature_history_mask[1, 1, 2] == false  # push 2 (NaN)
        @test layer.feature_history_mask[1, 1, 3] == true   # push 3
    end
end
