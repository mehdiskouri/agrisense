using Test
using AgriSenseCore

@testset "AgriSenseCore" begin
    include("test_types.jl")
    include("test_hypergraph.jl")
    include("test_bridge.jl")
    include("test_irrigation.jl")
    include("test_synthetic.jl")
end
