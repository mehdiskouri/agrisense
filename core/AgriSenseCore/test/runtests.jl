using Test
using AgriSenseCore

# Allow scalar indexing on CuArrays during tests — tests validate values element-by-element
if AgriSenseCore.HAS_CUDA
    using CUDA
    CUDA.allowscalar(true)
end

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
    include("test_gpu.jl")
end
