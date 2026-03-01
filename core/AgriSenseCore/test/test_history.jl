using Test
using AgriSenseCore
using SparseArrays

# ---------------------------------------------------------------------------
@testset "Ring Buffer — push_features! & get_history" begin

    # Helper: create a minimal HyperGraphLayer for history testing
    function make_layer(nv::Int, d::Int; buf_size::Int=DEFAULT_HISTORY_SIZE)
        B = sparse(Int32[1], Int32[1], Float32[1.0], nv, 1)
        vf = zeros(Float32, nv, d)
        fh = zeros(Float32, nv, d, buf_size)
        return HyperGraphLayer(B, vf, fh, 1, 0,
                               [Dict{String,Any}()],
                               ["v$i" for i in 1:nv],
                               ["e1"])
    end

    @testset "empty buffer returns empty matrix" begin
        layer = make_layer(2, 3)
        hist = get_history(layer, 1)
        @test size(hist) == (3, 0)
    end

    @testset "push single reading" begin
        layer = make_layer(2, 3)
        push_features!(layer, 1, Float32[1.0, 2.0, 3.0])

        # Snapshot updated
        @test layer.vertex_features[1, :] == Float32[1.0, 2.0, 3.0]

        # History has 1 entry
        @test layer.history_length == 1
        hist = get_history(layer, 1)
        @test size(hist) == (3, 1)
        @test hist[:, 1] == Float32[1.0, 2.0, 3.0]
    end

    @testset "push multiple readings in order" begin
        layer = make_layer(1, 2; buf_size=10)
        for t in 1:5
            push_features!(layer, 1, Float32[Float32(t), Float32(t * 10)])
        end
        @test layer.history_length == 5
        hist = get_history(layer, 1)
        @test size(hist) == (2, 5)
        # Oldest-first
        @test hist[1, :] == Float32[1.0, 2.0, 3.0, 4.0, 5.0]
        @test hist[2, :] == Float32[10.0, 20.0, 30.0, 40.0, 50.0]
    end

    @testset "ring buffer wraps correctly" begin
        buf_size = 4
        layer = make_layer(1, 1; buf_size=buf_size)

        # Push 6 values into buffer of size 4
        for t in 1:6
            push_features!(layer, 1, Float32[Float32(t)])
        end

        @test layer.history_length == buf_size
        hist = get_history(layer, 1)
        @test size(hist) == (1, buf_size)

        # After wrapping, oldest should be 3, not 1
        # Buffer contains [3, 4, 5, 6] in oldest-first order
        @test hist[1, :] == Float32[3.0, 4.0, 5.0, 6.0]
    end

    @testset "current snapshot reflects last push" begin
        layer = make_layer(3, 2; buf_size=5)
        push_features!(layer, 2, Float32[7.0, 8.0])
        push_features!(layer, 2, Float32[9.0, 10.0])
        # Snapshot should be the last push
        @test layer.vertex_features[2, :] == Float32[9.0, 10.0]
        # Vertex 1 untouched
        @test layer.vertex_features[1, :] == Float32[0.0, 0.0]
    end

    @testset "partial features (shorter than d)" begin
        layer = make_layer(1, 4; buf_size=3)
        # Push only 2 features into a 4-dim layer
        push_features!(layer, 1, Float32[1.0, 2.0])
        @test layer.vertex_features[1, 1:2] == Float32[1.0, 2.0]
        @test layer.vertex_features[1, 3:4] == Float32[0.0, 0.0]
    end

    @testset "serialization round-trip preserves history" begin
        # Build a small graph with history
        config = Dict{String,Any}(
            "farm_id" => "hist-test",
            "farm_type" => "greenhouse",
            "active_layers" => ["soil"],
            "zones" => [],
            "models" => Dict("irrigation" => true, "nutrients" => false,
                             "yield_forecast" => false, "anomaly_detection" => false),
            "vertices" => [Dict{String,Any}("id" => "v1", "type" => "sensor"),
                           Dict{String,Any}("id" => "v2", "type" => "sensor")],
            "edges" => [Dict{String,Any}("id" => "e1", "layer" => "soil",
                        "vertex_ids" => ["v1", "v2"],
                        "metadata" => Dict{String,Any}())],
        )
        state = build_graph(config)
        graph = deserialize_graph(state)

        # Push some history
        push_features!(graph.layers[:soil], 1, Float32[0.3, 25.0, 1.2, 6.5])
        push_features!(graph.layers[:soil], 1, Float32[0.28, 24.5, 1.1, 6.4])

        # Serialize then deserialize
        state2 = serialize_graph(graph)
        graph2 = deserialize_graph(state2)

        @test graph2.layers[:soil].history_length == 2
        @test graph2.layers[:soil].history_head == graph.layers[:soil].history_head
        hist1 = get_history(graph.layers[:soil], 1)
        hist2 = get_history(graph2.layers[:soil], 1)
        @test hist1 ≈ hist2
    end
end

# ===========================================================================
# Phase 13 — get_history return_mask Tests
# ===========================================================================
@testset "Phase 13 — get_history with return_mask" begin

    # Helper (same as above)
    function make_layer_p13(nv::Int, d::Int; buf_size::Int=DEFAULT_HISTORY_SIZE)
        B = sparse(Int32[1], Int32[1], Float32[1.0], nv, 1)
        vf = zeros(Float32, nv, d)
        fh = zeros(Float32, nv, d, buf_size)
        return HyperGraphLayer(B, vf, fh, 1, 0,
                               [Dict{String,Any}()],
                               ["v$i" for i in 1:nv],
                               ["e1"])
    end

    @testset "return_mask=false gives Matrix only" begin
        layer = make_layer_p13(1, 2; buf_size=5)
        push_features!(layer, 1, Float32[1.0, 2.0])
        result = get_history(layer, 1; return_mask=false)
        @test result isa Matrix{Float32}
    end

    @testset "return_mask=true gives (Matrix, BoolMatrix) tuple" begin
        layer = make_layer_p13(1, 2; buf_size=5)
        push_features!(layer, 1, Float32[1.0, 2.0])
        data, mask = get_history(layer, 1; return_mask=true)
        @test data isa Matrix{Float32}
        @test mask isa Matrix{Bool}
        @test size(data) == size(mask)
    end

    @testset "mask reflects NaN validity" begin
        layer = make_layer_p13(1, 3; buf_size=5)
        push_features!(layer, 1, Float32[1.0, NaN, 3.0])
        push_features!(layer, 1, Float32[NaN, 2.0, NaN])

        data, mask = get_history(layer, 1; return_mask=true)
        @test size(data) == (3, 2)
        @test size(mask) == (3, 2)

        # Push 1: [1.0, NaN, 3.0] → mask col 1 = [true, false, true]
        @test mask[1, 1] == true
        @test mask[2, 1] == false
        @test mask[3, 1] == true

        # Push 2: [NaN, 2.0, NaN] → mask col 2 = [false, true, false]
        @test mask[1, 2] == false
        @test mask[2, 2] == true
        @test mask[3, 2] == false
    end

    @testset "mask preserves order after ring buffer wrap" begin
        buf_size = 3
        layer = make_layer_p13(1, 1; buf_size=buf_size)
        # Push: valid, NaN, valid, NaN (wraps)
        push_features!(layer, 1, Float32[1.0])
        push_features!(layer, 1, Float32[NaN])
        push_features!(layer, 1, Float32[3.0])
        push_features!(layer, 1, Float32[NaN])  # overwrites slot 1

        data, mask = get_history(layer, 1; return_mask=true)
        @test size(data) == (1, 3)
        # After wrap, oldest-first order: slot2(NaN), slot3(valid), slot1(NaN)
        @test mask[1, 1] == false  # oldest: was NaN push
        @test mask[1, 2] == true   # middle: valid 3.0
        @test mask[1, 3] == false  # newest: NaN
    end

    @testset "empty buffer with return_mask" begin
        layer = make_layer_p13(1, 2; buf_size=5)
        data, mask = get_history(layer, 1; return_mask=true)
        @test size(data) == (2, 0)
        @test size(mask) == (2, 0)
    end
end
