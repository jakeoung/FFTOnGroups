This repository aims to implement Fourier transform on SE(2) group.

Please check `examples` folder.

Usage example:

```
# # Example: Fourier transform on SE(2)

using Plots
using Statistics
# using FFTOnGroups
using DrWatson
@quickactivate "FFTOnGroups"

n01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))

# Test1: check the invertability
f_test = zeros(ComplexF64, 50,50,90)
f_test[10:21, 11:30, 1:10] .= 1.0 + 1.0im
# f_test[11, 15:25, 3:33] .= 2.0 + 2.0im
H, W = size(f_test)

option = 1
bextend=true

if option == 1
    include(srcdir("SE2.jl"))

    fhat0_test, pp, mm, nn = fft_SE2(f_test; bextend=bextend) # fhat(m,n,p)
    @time f1_adjoint = adjoint_fft_SE2(fhat0_test, H, W; bextend=bextend);
    @time f1 = ifft_SE2(fhat0_test, H, W; bextend=bextend);

    @show mean(abs.(f1_adjoint .- f_test))
    @show mean(abs.(f1 .- f_test))
elseif option == 2
    include(srcdir("SE2_v0_polar.jl"))

    fhat0_test, pp, phis, mm, nn, At, coord = fft_SE2_polar(f_test) # fhat(m,n,p)
    f1_adjoint = ifft_polar0(fhat0_test, mm, At, H, W, coord)
    @show mean(abs.(f1_adjoint .- f_test))
    f1 = f1_adjoint

elseif option == 3
    # f_test = zeros(ComplexF64, 51,51,200)
    # f_test[10:21, 11:30, :] .= 1.0 + 1.0im
    # f_test[11, 15:25, :] .= 2.0 + 2.0im
    # H, W = size(f_test)

    include(srcdir("SE2_polar.jl"))
    fhat, pp, mm, nn = polar_fft_SE2(f_test)
    f1_adjoint = polar_ifft_SE2(fhat, H, W, mm, nn)

    @show mean(abs.(f1_adjoint .- f_test))
end

# check the independence of theta
# @show unique(fhat0_test[:,10,40])

# plot as images
plot(heatmap(real.(f_test[:,:,1]), title="original"),
    heatmap(real.(f1_adjoint[:,:,1]), title="adjoint(fft(f))"),
    heatmap(real.(f1[:,:,1]), title="ifft(fft(f))"))

# plot profiles for the single column
plot(imag.(f_test[11,:,1]), label="original"); plot!(imag.(f1_adjoint[11,:,1]), label="adjoint_fft")
plot!(imag.(35*f1[11,:,1]), label="ifft")

```
