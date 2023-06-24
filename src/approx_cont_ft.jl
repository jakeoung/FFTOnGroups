using FFTW

"""
    approx_cont_ft(x, y) → z
Dummy function for illustration purposes.
Performs operation:
```math
z = x + y
```

# [link 1](https://math.stackexchange.com/questions/377073/calculate-the-fourier-transform-of-bx-frac1x2-a2)
# [link 2](https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.01%3A_Conversion_of_Continuous_Fourier_Transform_to_DFT)
# [link 3](https://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy)
"""
function approx_cont_ft(f::Array{T, 1}, x0, dx) where {T}
    N = length(f)
    # kn = 2 * pi / (N*dx) .* collect(0:N-1)
    kn = 2*pi / dx * fftfreq(N, 1)
    dft = fft(f)
    fhat = dx * dft .* exp.(-im * kn * x0)
    return fhat, kn
end

function approx_cont_ft(f::Array{T, 2}, x0, dx, y0, dy) where {T}
    # x: row, y: column
    N1, N2  = size(f)
    
    kn1 = 2*pi / dx .* fftfreq(N1, 1)
    kn2 = 2*pi / dy .* fftfreq(N2, 1)
    freqs = [kn1, kn2]

    dft = fft(f)
    phases = reshape(exp.(-im * kn1 * x0), N1,1) .* reshape(exp.(-im * kn2 * y0), 1, N2)
    fhat = dx*dy * dft .* phases
    return fhat, freqs
end
