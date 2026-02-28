using Test
using AgriSenseCore

@testset "Irrigation Scheduler" begin
    @testset "schedule returns Vector{Dict}" begin
        state = Dict{String,Any}(
            "farm_id" => "farm-001",
            "n_vertices" => 0,
            "vertex_index" => Dict{String,Int}(),
            "layers" => Dict{String,Any}(),
        )
        result = AgriSenseCore.irrigation_schedule(state, 7)
        @test result isa Vector{Dict{String,Any}}
    end
end
