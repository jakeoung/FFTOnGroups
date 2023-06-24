# # Example: Approximate continuous Fourier transform by DFT

using Plots

function cart2polar(f)
    
end

#--------------------------------------------------
# Test
#--------------------------------------------------
# [ref 1](https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/spectral/fourier_interpolation.py)

A = [1, 2, 3, 4, 5]

Ahat = fft(A)

out = zeros(ComplexF64, length(A))
out_neg = zeros(ComplexF64, length(A))
for k = 1 : length(A)
    for n = 1:length(A)
        out[k] += exp(-im * 2*pi*(n-1)*(k-1) / length(A)) * A[n]
        out_neg[k] += exp(-im * 2*pi*(n-1)*(-(k-1)) / length(A)) * A[n]
    end
end

@show Ahat
@show out
@show out_neg
@show fftfreq(5, 5)