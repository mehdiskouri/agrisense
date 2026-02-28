using Test
using AgriSenseCore

@testset "AgriSenseCore" begin
    include("test_hypergraph.jl")
    include("test_irrigation.jl")
    include("test_synthetic.jl")
end
