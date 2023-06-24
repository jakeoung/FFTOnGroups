# # Example: Fourier transform on SE(2)

using DrWatson, Test
using Plots

@quickactivate "FFTOnGroups"
n01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))
include(srcdir("SE3.jl"))

# Test1: check the invertability
f_test = zeros(ComplexF64, 50,50,50,80)
f_test[10:21, 11:30, 10:21, 21:30, :] .= 1.0 + 1.0im
f_test[11, 15:25, 10:21, 21:30, :] .= 2.0 + 2.0im
H, W = size(f_test)

fhat0_test, pp, mm, nn = fft_SE3(f_test) # fhat(m,n,p)
@time f1_adjoint = adjoint_fft_SE3(fhat0_test, H, W);
@time f1 = ifft_SE3(fhat0_test, H, W);

fhat0_test, pp, mm, nn = polar_fft_SE2(f_test) # fhat(m,n,p)
@time f1_polar = ifft_SE2(fhat0_test, H, W);
# @time f1_polar = polar_ifft_SE2(fhat0_test, H, W);

# plot as images
plot(heatmap(real.(f_test[:,:,1,1]), title="original"),
    heatmap(real.(f1_adjoint[:,:,1,1]), title="adjoint(fft(f))"),
    heatmap(real.(f1[:,:,1,1]), title="ifft(fft(f))"),)
    # heatmap(real.(f1_polar[:,:,1,1]), title="by polarfft (~ing)"))


# plot profiles for the single column
plot(real.(f_test[11,:,1,1]), label="original"); plot!(real.(f1_adjoint[11,:,1]), label="adjoint_fft"); plot!(real.(f1[11,:,1]), label="ifft")

# function test_conv()
#     # f_test = zeros(40,40,10)
#     # f_test[10:21, 11:13, :] .= 1.0
#     f_test = zeros(40,40,3)
#     f_test[10:15, 11:20, :] .= 1.0
#     thetas = LinRange(0, 2*pi-(2*pi)/size(f_test,3), size(f_test,3));
#     f2 = similar(f_test)
#     f2 .= 0.1
#     f_conv = SE2_conv(f_test, f2, thetas)
#     # @show sum(abs.(f_test .- f1)) / length(f_test)
#     save("../result/test_SE2_conv0.png", n01(abs.(f_test[:,:,2])))
#     save("../result/test_SE2_conv.png", n01(abs.(f_conv[:,:,2])))
# end