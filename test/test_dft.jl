# # Example: Approximate continuous Fourier transform by DFT

using Plots

function cart2polar(f)
    
end

#--------------------------------------------------
# Test
#--------------------------------------------------
# [ref 1](https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/spectral/fourier_interpolation.py)

make_grid(x, y) = repeat(x', length(y), 1), repeat(y, 1, length(x))
xx = collect(-0.5:0.02:0.5)
yy = collect(-0.5:0.02:0.5)

XX, YY = make_grid(xx, yy)

f = exp.(2*pi*im*(XX .+ 0.5))

f_polar = 
f_cart = 