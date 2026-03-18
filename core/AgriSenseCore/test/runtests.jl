using Test
using AgriSenseCore

function _run_all_tests()::Nothing
    @testset "AgriSenseCore" begin
        include("test_types.jl")
        include("test_hypergraph.jl")
        include("test_bridge.jl")
        include("test_history.jl")
        include("test_irrigation.jl")
        include("test_nutrients.jl")
        include("test_yield.jl")
        include("test_backtesting.jl")
        include("test_anomaly.jl")
        include("test_synthetic.jl")
        include("test_gpu.jl")
    end
    return nothing
end

# Scope scalar indexing opt-in to this test run only when CUDA is available.
if AgriSenseCore.HAS_CUDA
    using CUDA
    CUDA.allowscalar() do
        _run_all_tests()
    end
else
    _run_all_tests()
end
