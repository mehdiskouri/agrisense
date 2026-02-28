# ---------------------------------------------------------------------------
# Cross-layer correlation injection for synthetic data
# ---------------------------------------------------------------------------

const SYNTHETIC_DROPOUT_RATE = 0.03f0

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

"""
    synthetic_steps(days; cadence_minutes=CADENCE_MINUTES) -> Int

Number of synthetic timesteps for a given horizon.
"""
function synthetic_steps(days::Int; cadence_minutes::Int=CADENCE_MINUTES)::Int
    return Int(days * 24 * 60 ÷ cadence_minutes)
end

"""
    time_grid(n_steps, cadence_minutes) -> Vector{Float32}

Returns a monotonically increasing time grid in hours.
"""
function time_grid(n_steps::Int, cadence_minutes::Int)::Vector{Float32}
    step_h = Float32(cadence_minutes) / 60.0f0
    return Float32.((0:n_steps-1) .* step_h)
end

"""Allocate zeros on requested backend."""
function backend_zeros(::Type{T}, dims...; use_gpu::Bool=false) where T
    if use_gpu && HAS_CUDA
        return CUDA.zeros(T, dims...)
    end
    return zeros(T, dims...)
end

"""Standard normal samples on requested backend."""
function backend_randn(::Type{T}, dims...; seed::Int=42, use_gpu::Bool=false) where T
    if use_gpu && HAS_CUDA
        CUDA.seed!(seed)
        return CUDA.randn(T, dims...)
    end
    rng = MersenneTwister(seed)
    return randn(rng, T, dims...)
end

"""
    stabilize_psd(C) -> Matrix{Float32}

Adds minimal diagonal jitter so Cholesky succeeds on near-singular matrices.
"""
function stabilize_psd(C::AbstractMatrix{Float32})::Matrix{Float32}
    Ccpu = Matrix{Float32}(C)
    jitter = 1.0f-5
    for _ in 1:6
        S = Symmetric(Ccpu + jitter * I)
        ok = true
        try
            cholesky(S)
        catch
            ok = false
        end
        ok && return Matrix{Float32}(S)
        jitter *= 10.0f0
    end
    return Matrix{Float32}(Symmetric(Ccpu + 1.0f-1 * I))
end

"""
    correlated_noise(n_steps, n_channels; seed, use_gpu) -> AbstractMatrix{Float32}

Produces `n_steps × n_channels` Gaussian noise with channel correlation.
CPU path uses BLAS; GPU path uses CuArray matmul.
"""
function correlated_noise(n_steps::Int, n_channels::Int;
                          seed::Int=42,
                          use_gpu::Bool=false,
                          corr_matrix::Union{Nothing, AbstractMatrix{Float32}}=nothing)
    C = corr_matrix === nothing ?
        build_correlation_matrix(n_channels; seed=seed + 700) : Matrix{Float32}(corr_matrix)
    C = stabilize_psd(Float32.(C))

    if use_gpu && HAS_CUDA
        Z = backend_randn(Float32, n_steps, n_channels; seed=seed + 701, use_gpu=true)
        L_cpu = Matrix{Float32}(cholesky(Symmetric(C)).L)
        L_gpu = CuArray(L_cpu)
        return Z * transpose(L_gpu)
    end

    rng = MersenneTwister(seed + 701)
    Z = randn(rng, Float32, n_steps, n_channels)
    L = Matrix{Float32}(cholesky(Symmetric(C)).L)
    return Z * transpose(L)
end

"""
    apply_dropout_with_mask(values; rate, seed, use_gpu) -> (values, mask)

Injects NaNs according to Bernoulli dropout and returns a boolean missingness mask.
"""
function apply_dropout_with_mask(values::AbstractMatrix{Float32};
                                 rate::Float32=SYNTHETIC_DROPOUT_RATE,
                                 seed::Int=42,
                                 use_gpu::Bool=false)
    n_steps, n_channels = size(values)
    mask = if use_gpu && HAS_CUDA
        CUDA.seed!(seed + 911)
        CUDA.rand(Float32, n_steps, n_channels) .< rate
    else
        rng = MersenneTwister(seed + 911)
        rand(rng, Float32, n_steps, n_channels) .< rate
    end
    with_nan = ifelse.(mask, Float32(NaN), values)
    return with_nan, mask
end

"""Bring arrays to plain CPU arrays for bridge-safe Dict outputs."""
function cpu_plain(arr)
    return Array(ensure_cpu(arr))
end

"""Convert an array to active backend with Float32 element type."""
function to_backend(arr; use_gpu::Bool=false)
    if use_gpu && HAS_CUDA
        return CuArray(Float32.(arr))
    end
    return Float32.(arr)
end

