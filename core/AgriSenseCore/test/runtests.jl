using Test
using AgriSenseCore

@testset "AgriSenseCore" begin
    include("test_types.jl")
    include("test_hypergraph.jl")
    include("test_bridge.jl")
    include("test_history.jl")
    include("test_irrigation.jl")
    include("test_nutrients.jl")
    include("test_yield.jl")
    include("test_anomaly.jl")
    include("test_synthetic.jl")
end
