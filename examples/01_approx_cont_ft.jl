# # Example: Approximate continuous Fourier transform by DFT

using Plots, FFTW

function approx_cont_ft(f::Array{T, 1}, x0, dx) where {T}
    N = length(f)
    
    kn = 2*pi / dx * fftfreq(N, 1)
    dft = fft(f)
    fhat = dx * dft .* exp.(-im * kn * x0)

    return fhat, kn
end

#--------------------------------------------------
# Test F.T. of $1 / (x^2+ 1)$
# the analytic solution is $\hat f(w) = \pi * \exp(-|w|)$
#--------------------------------------------------


dx = 0.1
xx = collect(-100:dx:100)
(mod(length(xx), 2) == 1) && (xx = xx[1:end-1])

fx = 1.0 ./ (xx.^2 .+ 1.0)
fhat, kn = approx_cont_ft(fx, xx[1], dx)

fhat_trues = pi * exp.( - abs.(kn) )

plot(abs.(fhat), linestyle=:dash, label="est")
plot!(fhat_trues, label="true")

#--------------------------------------------------
# Test f(x) = \int \hat f(w) exp(iwx) dw
# where $\hat f(w) = \pi * \exp(-|w|)$
# analytic solution is 2pi / (x^2+1) 
# IFT(fx) = conj(fft(conj(fx)))
#--------------------------------------------------

kn_shift = fftshift(kn)
fhat_inv, fhat_inv_xx = approx_cont_ft(conj.(fhat), kn_shift[1], kn_shift[2]-kn_shift[1])
fhat_inv = fftshift(conj(fhat_inv)) 
fhat_inv_xx = fftshift(fhat_inv_xx)

f_trues = 2*pi ./ (fhat_inv_xx.^2 .+ 1.0) 

plot(abs.(fhat_inv), linestyle=:dash, label="est")
plot!(f_trues,  linestyle=:dashdot, label="true")
