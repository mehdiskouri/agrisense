# ---------------------------------------------------------------------------
# Cross-layer correlation injection for synthetic data
# ---------------------------------------------------------------------------

const SYNTHETIC_DROPOUT_RATE = 0.03f0

# ---------------------------------------------------------------------------
# Contiguous outage injection constants
# ---------------------------------------------------------------------------

"""Per-sensor per-timestep probability that a contiguous outage *starts*."""
const DEFAULT_OUTAGE_PROB = 0.005f0

"""(min, max) outage duration in timesteps — 4..96 at 15-min cadence = 1..24 hours."""
const DEFAULT_OUTAGE_DURATION_RANGE = (4, 96)

"""Named-tuple field schema for outage event metadata."""
const OUTAGE_EVENT_FIELDS = (:channel, :start, :duration)

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

# ---------------------------------------------------------------------------
# Contiguous outage injection — CPU algorithm (threaded per-channel)
# ---------------------------------------------------------------------------

"""
    apply_contiguous_outage_cpu!(mask, values; outage_prob, duration_range, seed)
        -> Vector{@NamedTuple{channel::Int, start::Int, duration::Int}}

CPU path: per-channel sequential walk with `Threads.@threads` parallelism across
channels.  Each channel gets a deterministic per-channel RNG seeded from
`seed + 1337 + col`, so output is bitwise-reproducible regardless of thread count.

Mutates `mask` and `values` in-place, stamping contiguous `true` / `NaN` blocks.
Events within each channel are non-overlapping by construction (the walk skips
past the end of each outage).
"""
function apply_contiguous_outage_cpu!(mask::BitMatrix,
                                      values::Matrix{Float32};
                                      outage_prob::Float32=DEFAULT_OUTAGE_PROB,
                                      duration_range::Tuple{Int,Int}=DEFAULT_OUTAGE_DURATION_RANGE,
                                      seed::Int=42,
                                      )::Vector{@NamedTuple{channel::Int, start::Int, duration::Int}}
    n_steps, n_channels = size(values)
    dmin, dmax = duration_range

    # Per-channel event buffers — filled in parallel, merged after
    per_channel_events = Vector{Vector{@NamedTuple{channel::Int, start::Int, duration::Int}}}(undef, n_channels)

    Threads.@threads for col in 1:n_channels
        # Deterministic per-channel RNG: seed is channel-dependent so
        # output is identical regardless of thread scheduling / thread count.
        ch_rng = MersenneTwister(seed + 1337 + col)
        ch_events = @NamedTuple{channel::Int, start::Int, duration::Int}[]
        t = 1
        while t <= n_steps
            if rand(ch_rng, Float32) < outage_prob
                dur = rand(ch_rng, dmin:dmax)
                dur = min(dur, n_steps - t + 1)          # clamp to remaining rows
                t_end = t + dur - 1
                @inbounds for r in t:t_end
                    mask[r, col] = true
                    values[r, col] = Float32(NaN)
                end
                push!(ch_events, (channel=col, start=t, duration=dur))
                t = t_end + 1                             # skip past outage
            else
                t += 1
            end
        end
        per_channel_events[col] = ch_events
    end

    # Merge per-channel results in deterministic channel order
    events = @NamedTuple{channel::Int, start::Int, duration::Int}[]
    for col in 1:n_channels
        append!(events, per_channel_events[col])
    end
    return events
end

# ---------------------------------------------------------------------------
# Contiguous outage injection — GPU scatter-write kernel
# ---------------------------------------------------------------------------

"""
    outage_stamp_kernel!(mask_gpu, values_gpu, channels, starts, durations)

GPU kernel: each thread stamps one outage event.  Thread `i` writes
`mask[start[i]:start[i]+dur[i]-1, channel[i]] = true` and the corresponding
slots in `values` to `NaN32`.  No atomics required — events within a channel
are non-overlapping by construction from the CPU sampling phase.
"""
@kernel function outage_stamp_kernel!(mask_gpu, values_gpu,
                                      @Const(channels), @Const(starts),
                                      @Const(durations), n_steps::Int32)
    i = @index(Global)
    @inbounds begin
        col   = channels[i]
        t0    = starts[i]
        dur   = durations[i]
        t_end = min(t0 + dur - Int32(1), n_steps)
        for t in t0:t_end
            mask_gpu[t, col]   = true
            values_gpu[t, col] = Float32(NaN)
        end
    end
end

# ---------------------------------------------------------------------------
# Contiguous outage injection — CPU/GPU dispatcher
# ---------------------------------------------------------------------------

"""
    apply_contiguous_outage!(mask, values; outage_prob, duration_range, seed, use_gpu)
        -> (outage_events, outage_mask)

Inject contiguous outage blocks into `mask` and `values`.  Returns:
  - `outage_events` — structured list of `(channel, start, duration)` tuples
  - `outage_mask`   — BitMatrix recording *only* the outage-caused entries
                      (a subset of the combined `mask`)

**GPU path** (two-phase):
  1. CPU pre-samples all outage events (serial per-channel, channels independent).
  2. GPU kernel scatter-writes the NaN/true stamps in parallel across events.

**CPU path**: calls `apply_contiguous_outage_cpu!` directly.
"""
function apply_contiguous_outage!(mask, values;
                                   outage_prob::Float32=DEFAULT_OUTAGE_PROB,
                                   duration_range::Tuple{Int,Int}=DEFAULT_OUTAGE_DURATION_RANGE,
                                   seed::Int=42,
                                   use_gpu::Bool=false)
    n_steps, n_channels = size(values)
    outage_mask = falses(n_steps, n_channels)

    if use_gpu && HAS_CUDA
        # --- Phase 1 (CPU): sample outage events ---
        # Work on CPU copies for the sampling walk
        mask_cpu   = BitMatrix(cpu_plain(mask))
        values_cpu = Matrix{Float32}(cpu_plain(values))
        events = apply_contiguous_outage_cpu!(mask_cpu, values_cpu;
                                              outage_prob=outage_prob,
                                              duration_range=duration_range,
                                              seed=seed)

        # Build outage_mask from events
        for e in events
            outage_mask[e.start:min(e.start + e.duration - 1, n_steps), e.channel] .= true
        end

        if isempty(events)
            # Nothing to stamp — copy CPU-modified mask/values back
            mask  .= CuArray(mask_cpu)
            values .= CuArray(values_cpu)
            return events, outage_mask
        end

        # --- Phase 2 (GPU): scatter-write outage stamps ---
        ch_arr  = CuArray(Int32[e.channel  for e in events])
        st_arr  = CuArray(Int32[e.start    for e in events])
        dur_arr = CuArray(Int32[e.duration for e in events])

        # We must stamp onto the *device-resident* mask and values that the
        # caller owns.  First re-apply the Bernoulli mask (already there),
        # then overlay the outage stamps via the kernel.
        launch_kernel!(outage_stamp_kernel!, CUDABackend(), length(events),
                       mask, values, ch_arr, st_arr, dur_arr, Int32(n_steps))

        return events, outage_mask
    else
        # --- Pure CPU path ---
        mask_bm   = BitMatrix(mask)
        values_m  = Matrix{Float32}(values)
        events = apply_contiguous_outage_cpu!(mask_bm, values_m;
                                              outage_prob=outage_prob,
                                              duration_range=duration_range,
                                              seed=seed)
        # Write results back into the caller's arrays
        mask  .= mask_bm
        values .= values_m

        # Build outage_mask from events
        for e in events
            outage_mask[e.start:min(e.start + e.duration - 1, n_steps), e.channel] .= true
        end

        return events, outage_mask
    end
end

# ---------------------------------------------------------------------------
# Bernoulli dropout + contiguous outage composition
# ---------------------------------------------------------------------------

"""
    apply_dropout_with_mask(values; rate, seed, use_gpu,
                            outage_prob, outage_duration_range)
        -> (values, mask, outage_events, outage_mask)

Injects missingness into a `(n_steps, n_channels)` matrix via two mechanisms:

1. **Bernoulli point dropout** — each cell independently dropped with probability `rate`.
2. **Contiguous block outages** — sensor failures producing NaN runs of configurable
   duration, triggered with probability `outage_prob` per timestep per channel.

Returns a 4-tuple:
  - `values`         — Float32 matrix with NaN at all dropout + outage positions
  - `mask`           — combined BitMatrix (Bernoulli ∪ outage)
  - `outage_events`  — `Vector{@NamedTuple{channel, start, duration}}` (empty if no outages)
  - `outage_mask`    — BitMatrix of *only* outage-caused entries
"""
function apply_dropout_with_mask(values::AbstractMatrix{Float32};
                                 rate::Float32=SYNTHETIC_DROPOUT_RATE,
                                 seed::Int=42,
                                 use_gpu::Bool=false,
                                 outage_prob::Float32=0.0f0,
                                 outage_duration_range::Tuple{Int,Int}=DEFAULT_OUTAGE_DURATION_RANGE)
    n_steps, n_channels = size(values)

    # --- Step 1: Bernoulli point dropout ---
    mask = if use_gpu && HAS_CUDA
        CUDA.seed!(seed + 911)
        CUDA.rand(Float32, n_steps, n_channels) .< rate
    else
        rng = MersenneTwister(seed + 911)
        rand(rng, Float32, n_steps, n_channels) .< rate
    end
    with_nan = ifelse.(mask, Float32(NaN), values)

    # --- Step 2: Contiguous outage overlay (if enabled) ---
    if outage_prob > 0.0f0
        outage_events, outage_mask = apply_contiguous_outage!(
            mask, with_nan;
            outage_prob=outage_prob,
            duration_range=outage_duration_range,
            seed=seed,
            use_gpu=use_gpu)
        return with_nan, mask, outage_events, outage_mask
    end

    # No outages requested — return empty event list and all-false outage mask
    empty_events = @NamedTuple{channel::Int, start::Int, duration::Int}[]
    empty_outage_mask = falses(n_steps, n_channels)
    return with_nan, mask, empty_events, empty_outage_mask
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

