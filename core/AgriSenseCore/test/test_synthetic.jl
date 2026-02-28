using Test
using AgriSenseCore

@testset "Synthetic Data Generator" begin
    @testset "generate_synthetic returns Dict with metadata" begin
        result = AgriSenseCore.generate_synthetic(farm_type="greenhouse", days=90, seed=42)
        @test result["farm_type"] == "greenhouse"
        @test result["days"] == 90
        @test result["seed"] == 42
    end
end
