# ---------------------------------------------------------------------------
# Cross-layer correlation injection for synthetic data
# ---------------------------------------------------------------------------

"""
    build_correlation_matrix(n_channels::Int; seed::Int=42) -> Matrix{Float32}

Build a positive-definite correlation matrix for cross-sensor correlation injection.
Uses a random Cholesky factor approach.
"""
function build_correlation_matrix(n_channels::Int; seed::Int=42)::Matrix{Float32}
    rng = MersenneTwister(seed)
    A = randn(rng, Float32, n_channels, n_channels)
    C = A * A'
    D = Diagonal(1.0f0 ./ sqrt.(diag(C)))
    return D * C * D  # normalize to correlation matrix
end
