using Manifolds
using LinearAlgebra
using Interpolations
import ScatteredInterpolation
using FFTW
using CoordinateTransformations

# using NFFT3


make_grid(x, y) = repeat(x', length(y), 1), repeat(y, 1, length(x))

function approx_cont_ft_SE2(f, x0=0.0, dx=1.0, y0=0.0, dy=1.0; use_ifft=false)
    N1, N2, ntheta  = size(f)
    
    kn1 = 2*pi / dx .* fftfreq(N1, 1)
    kn2 = 2*pi / dy .* fftfreq(N2, 1)
    # kn1 = fftfreq(N1, 1)
    # kn2 = fftfreq(N2, 1)

    phases = reshape(exp.(-im * kn1 * x0), N1,1) .* reshape(exp.(-im * kn2 * y0), 1, N2)

    if use_ifft == false
        fhat = dx*dy * fft(f, 1:2) .* phases
        # fhat = fftshift(dx*dy * dft .* phases, 1:2)
    else
        # dft = ifft(ifftshift(f, 1:2), 1:2)
        dft = ifft(f, 1:2)
        fhat = dx*dy * dft .* phases
    end

    return fftshift(fhat, (1,2)), (fftshift(kn1), fftshift(kn2))
end

function convert_c2polar(f, freqs, npsi)
    # See http://juliamath.github.io/Interpolations.jl/latest/control/
    # 

    H, W, ntheta = size(f)
    n_samples_r = H
    # freq_max = abs(minimum(freqs[1]))
    freq_max = abs(minimum(freqs[1])) * 1.4
    # r_max = sqrt(2*freq_max^2 + freq_max^2)
    r_max = freq_max 
    # r_max = 100

    r = LinRange(0.0, r_max, n_samples_r) #endpoint=true
    t = LinRange(0, 2*pi, npsi+1)[1:end-1]

    # meshgrid (ij)
    rr = ones(length(t))' .* r  # = pp
    tt = t' .* ones(length(r))

    r_sz = length(r)
    t_sz = length(t)

    f_polar_real = zeros(r_sz, t_sz, ntheta)
    f_polar_im = zeros(r_sz, t_sz, ntheta)

    X = rr .* cos.(tt)
    Y = rr .* sin.(tt)

    for k = 1 : ntheta
        itp = interpolate(real(f[:,:,k]), BSpline(Cubic(Flat(OnGrid()))))
        sitp = scale(itp, freqs[1], freqs[2])
        sitp = extrapolate(sitp, Flat())
        # sitp = extrapolate(sitp, Line())
        f_polar_real[:,:,k] = sitp.(Y, X)

        itp = interpolate(imag(f[:,:,k]), BSpline(Cubic(Flat(OnGrid()))))
        sitp = scale(itp, freqs[1], freqs[2])
        sitp = extrapolate(sitp, Flat())
        # sitp = extrapolate(sitp, Line())
        f_polar_im[:,:,k] = sitp.(Y, X)
    end
    
    return f_polar_real .+ im .* f_polar_im, r, t
end

function map_coordinates(img, coord)
    out = similar(coord[1])
    itp = Interpolations.interpolate(img, BSpline(Linear()), OnCell())
    etpf = extrapolate(itp, Line())   # gives 1 on the left edge and 7 on the right edge
    # etpf = extrapolate(itp, Flat())   # gives 1 on the left edge and 7 on the right edge

    for i=1:size(out,1), j=1:size(out,2)
        # @show coord[1][i,j], coord[2][i,j]
        out[i,j] = etpf[coord[1][i,j], coord[2][i,j]]
    end
    return out
end


function wrap_coord(f, coord)
    H, W, C = size(f)
    f_wrap = zeros(typeof(f[1,1,1]), H+1, W+1, C)
    f_wrap[1:end-1,1:end-1,:] = f
    f_wrap[end,1:end-1,:] = f[1,:,:]
    f_wrap[1:end-1,end,:] = f[:,1,:]
    f_wrap[end,end,:] = f[1,1,:]

    coord[1] = mod.(coord[1], H)
    coord[2] = mod.(coord[2], W)

    coord[1] .+= 1
    coord[2] .+= 1

    return f, coord
end

function c2p_coords_spline(H, W, oversampling_factor = 1, npsi=nothing)
    p0 = H // 2
    q0 = W // 2

    spline_order = 1
    r_max = sqrt(p0^2 + q0^2)

    n_samples_r = oversampling_factor * (ceil(Int, r_max) + 1)
    if isnothing(npsi)
        npsi = oversampling_factor * (ceil(Int, 2*pi*r_max))
    end

    r = LinRange(0, r_max, n_samples_r) #endpoint=true
    t = LinRange(0, 2*pi, npsi+1)[1:end-1]

    # meshgrid (ij)
    rr = ones(length(t))' .* r
    tt = t' .* ones(length(r))

    X = rr .* cos.(tt)
    Y = rr .* sin.(tt)

    I = X .+ p0 
    J = Y .+ q0 
    return [I, J], r, t
end

function p2c_coords_spline(H, W, n_samples_r, npsi)
    p0 = H // 2
    q0 = W // 2
    #---------------------------------------------------
    # aaa
    #---------------------------------------------------
    i = 0:W-1
    j = 0:H-1
    x = i .- p0
    y = j .- q0

    xx = ones(length(y))' .* x
    yy = y' .* ones(length(y))

    R = sqrt.(xx.^2 + yy.^2)
    T = atan.(yy, xx)

    # if ~isnothing(n_samples_r)
        r_max = sqrt(p0^2 + q0^2)

        R .*= (n_samples_r - 1) / r_max
        # TODO (to understand)
        # The maximum angle in T is arbitrarily close to 2 pi,
        # but this should end up 1 pixel past the last index npsi - 1, i.e. it should end up at npsi
        # which is equal to index 0 since wraparound is used.
        T .*= npsi / (2 * pi)
    # end
    @show R
    # R .+= 1 # julia index starts from 1
    # T .+= 1
    return [R, T]
end

function convert_c2polar0(f, npsi=nothing)
    # warp(f, c2)
    H, W, ntheta = size(f)
    coord, rr, tt = c2p_coords_spline(H, W, 1, npsi)

    f_wrap, coord = wrap_coord(f, coord)

    r_sz = size(coord[1],1)
    t_sz = size(coord[1],2)
    f_polar_real = zeros(r_sz, t_sz, ntheta)
    f_polar_im = zeros(r_sz, t_sz, ntheta)
    for i = 1 : ntheta
        f_polar_real[:,:,i] = map_coordinates(real(f_wrap[:,:,i]), coord)
        f_polar_im[:,:,i] = map_coordinates(imag(f_wrap[:,:,i]), coord)
    end
    pp = rr
    return f_polar_real .+ im .* f_polar_im, pp, tt
end



@doc raw"""
    fft_SE2(f) -> fhat

Compute $\hat f (r,p,q)$ [m, n, p], m:theta, n:phi

# Arguments
- `f`: f [H, W, thetas]

"""
function fft_SE2(f; fft_method="cont_dft")
    npsi = size(f, 3)
    bnormalize = false

    #--------------------------------------------------
    # STEP 1. IFFT for f
    #--------------------------------------------------
    H, W, ntheta = size(f)        
    if fft_method == "cont_dft"
        f1s, freqs = approx_cont_ft_SE2(conj(f), 0.0, 2*pi/size(f,1), 0.0, 2*pi/size(f,2))
        f1s = conj(f1s)
    elseif fft_method == "fft"
        fs = fftshift(f, (1,2))
        f1s = conj(ifftshift(ifft(conj(fs), (1,2)), (1,2)) )
        freqs = (fftshift(fftfreq(H, 1)), fftshift(fftfreq(W, 1)))
    elseif fft_method == "nfft3"
        # https://nfft.github.io/NFFT3.jl/stable/index.html#PlonkaPottsSteidlTasche2018
    end
    
    # fs = f[y, x, :]
    # f1s = ifftshift(ifft(f, (1,2)), (1,2)) 

    #--------------------------------------------------
    # 2. Change cartesian to polar coordinates
    #--------------------------------------------------
    # f1p [r, phi, theta]
    f1p, pp, tt = convert_c2polar(f1s, freqs, npsi)
    # f1p, pp, tt = convert_c2polar0(f1s, npsi)

    mm = Int.(fftshift(fftfreq(size(f1p, 3), size(f1p, 3))))
    nn = Int.(fftshift(fftfreq(size(f1p, 2), size(f1p, 2))))
    
    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f2 = conj(fftshift(fft(conj(f1p), (3)), (3)))
    if bnormalize
        f2 *= ( 2*pi / ntheta ) 
    end

    # m_min = -floor(Int, size(f2, 3) / 2)    # m_max = ceil(Int, size(f1p, 3) / 2) -1 # @show m_min, m_max, mm

    psi = LinRange(0.0, 2*pi, size(f2, 2)+1)[1:end-1]
    factor = exp.(-im .* reshape(psi, 1,:,1) .* reshape(mm, 1,1,:))
    # @show size(factor), size(f2)
    f2 = f2 .* factor

    #--------------------------------------------------
    # 4. psi integration
    #--------------------------------------------------
    fhat = conj(fftshift( fft(conj(f2), (2)) , (2))) # / size(fhat, 2) # fhat(r,p,q)
    if bnormalize
        fhat *= (2*pi / size(f2, 2))
    end

    # [p, n, m] -> [m, n, p]
    fhat = permutedims(fhat, (3, 2, 1))

    return fhat, pp, tt, mm, nn, freqs
end

function convert_p2cart(fhat, r, t, freqs)
    # See http://juliamath.github.io/Interpolations.jl/latest/control/
    # https://github.com/arcaduf/gridding_method_minimal_oversampling/blob/master/gridding_module/pymodule_gridrec_v4/gridrec_v4_forwproj.c

    # now (r, psi, theta) (p, n, m)
    r_sz, npsi, ntheta = size(fhat)

    # rr = ones(length(t))' .* r  # = pp
    # tt = t' .* ones(length(r))

    H = r_sz # to be changed
    W = H

    y = freqs[1]
    x = freqs[2]

    # yy = ones(length(y))' .* x
    # xx = y' .* ones(length(y))

    yy = ones(length(y))' .* x
    xx = y' .* ones(length(y))

    R = sqrt.(xx.^2 + yy.^2)
    T = atan.(yy, xx)
    
    f_real = zeros(r_sz, r_sz, ntheta)
    f_im = zeros(r_sz, r_sz, ntheta)

    for k = 1 : ntheta
        itp = interpolate(real(fhat[:,:,k]), BSpline(Cubic(Flat(OnGrid()))))
        sitp = scale(itp, r, t)
        sitp = extrapolate(sitp, Line())
        f_real[:,:,k] = sitp.(R, T)

        itp = interpolate(imag(fhat[:,:,k]), BSpline(Cubic(Flat(OnGrid()))))
        sitp = scale(itp, r, t)
        sitp = extrapolate(sitp, Line())
        f_im[:,:,k] = sitp.(R, T)
    end    
    
    return f_real .+ im * f_im
end

# function convert_p2cart(fhat, r, t, freqs)
#     # See http://juliamath.github.io/Interpolations.jl/latest/control/
#     # https://github.com/arcaduf/gridding_method_minimal_oversampling/blob/master/gridding_module/pymodule_gridrec_v4/gridrec_v4_forwproj.c

#     # now (r, psi, theta) (p, n, m)
#     r_sz, npsi, ntheta = size(fhat)

#     rr = ones(length(t))' .* r  # = pp
#     tt = t' .* ones(length(r))

#     H = r_sz # to be changed
#     W = H
#     p0 = H / 2.0 + 0.5; q0 = W / 2.0 + 0.5

#     i = 0:W-1
#     j = 0:H-1
#     x = i .- p0
#     y = j .- q0

#     xx = ones(length(y))' .* x
#     yy = y' .* ones(length(y))

#     R = sqrt.(xx.^2 + yy.^2)
#     T = atan.(yy, xx)

#     X = rr .* cos.(tt)
#     Y = rr .* sin.(tt)

#     grid_points = [Y[:] X[:]]'
#     # grid_freq = [freqs[1][:] freqs[2][:]]'
#     grid_freq_ = make_grid(freqs[1], freqs[2])
#     grid_freq = [grid_freq_[2][:] grid_freq_[1][:]]'

#     f_real = zeros(r_sz, r_sz, ntheta)
#     f_im = zeros(r_sz, r_sz, ntheta)

#     # Step1: Find cartesian nbd points for each polar point 
#     for k = 1 : ntheta
#         itp = ScatteredInterpolation.interpolate(ScatteredInterpolation.Multiquadratic(), grid_points, real(fhat[:,:,k])[:])
#         f_real[:,:,k] = reshape(ScatteredInterpolation.evaluate(itp, grid_freq), r_sz, r_sz)

#         itp = ScatteredInterpolation.interpolate(ScatteredInterpolation.Multiquadratic(), grid_points, imag(fhat[:,:,k])[:])
#         f_im[:,:,k] = reshape(ScatteredInterpolation.evaluate(itp, grid_freq), r_sz, r_sz)
#     end
    
#     return f_real .+ im * f_im
# end

# inp: fhat_ of shape [m, n, p] (theta, psi, r)
# returns f[H,W,ntheta]
function adjoint_fft_SE2(fhat_, p, t, freqs; mm=nothing, use_cont_ft=true)
    if isnothing(mm)
        @assert mod(size(fhat_, 1), 2) == 0
        mm = collect(-size(fhat_, 1)//2 : size(fhat_, 1)//2-1)
    end
    fhat = permutedims(fhat_, (3,2,1))
    ntheta = size(fhat, 3)
    # now (r, psi, theta) (p, n, m)

    bnormalize = 0

    #--------------------------------------------------
    # 4. psi integration
    #--------------------------------------------------
    f2f = conj(ifft(ifftshift(conj(fhat), (2)), (2)))
    if bnormalize == 1
        f2f = f2f / (2*pi / size(f2f, 2))
    elseif bnormalize == 2
        f2f = f2f * (2*pi / size(f2f, 2))
    end
    
    psi = LinRange(0.0, 2*pi, size(f2f, 2)+1)[1:end-1]
    factor = exp.(im .* reshape(psi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2f .* factor

    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f1p = conj(ifft(ifftshift(conj(f2), (3)), (3)))
    if bnormalize == 1
        f1p = f1p / ( 2*pi / ntheta )
    elseif bnormalize == 2
        f1p = f1p * ( 2*pi / ntheta )
    end
    # f1p = ifftshift(f1p, (3)) / size(f1p, 3)

    #--------------------------------------------------
    # 2. change polar to cartesian coordinates
    #--------------------------------------------------
    f1c = convert_p2cart(f1p, p, t, freqs)
    # f1c = convert_p2cart(f1p)
    fs = f1c
    
    #--------------------------------------------------
    # 1. IFFT for f
    #--------------------------------------------------
    # H = size(fhat,1); W = H;
    # p0 = Int(H // 2); q0 = Int(W // 2);
    # y = mod.(-p0:-p0+H-1, H) .+ 1
    # x = mod.(-q0:-q0+W-1, W) .+ 1

    # f1s = fft(fs, (1,2))

    # f1s = ifftshift(fs, (1,2))
    # f1s = conj(ifft(conj(f1s)))
    
    # fs = fftshift(f, (1,2))
    # f1s = ifftshift(ifft(fs, (1,2)), (1,2)) 

    # f = fft(fftshift(fs, (1,2)), (1,2)) # optional
    f = fft(fftshift(fs, (1,2)), (1,2)) # optional
    f = ifftshift(f, (1,2))


    
    return f
end

# inp: fhat_ of shape [m, n, p] (theta, psi, r)
# returns f[H,W,ntheta]
function ifft_SE2(fhat_, p, t, freqs; mm=nothing, use_cont_ft=true)
    if isnothing(mm)
        @assert mod(size(fhat_, 1), 2) == 0
        mm = fftshift(Int.(fftfreq(size(fhat_, 1), size(fhat_, 1))))
        nn = fftshift(Int.(fftfreq(size(fhat_, 2), size(fhat_, 2))))
    end

    ntheta = size(fhat_, 1)

    # fhat_: (m,n,p)
    ftilde = zeros(ComplexF64, length(p), length(t), length(t))
    for in = 1 : size(fhat_, 2)
        n = nn[in]
        for ip = 1 : length(p)
            for ipsi = 1 : size(fhat_, 2)
                ftilde[ip, ipsi, in] = sum(fhat_[in,:,ip] .* exp.(im* (n .- mm) * t[ipsi] ))
            end
        end
    end

    f1p = ftilde
    # f1p = permutedims(ftilde, (3,2,1))
    # now (r, psi, theta) (p, n, m)

    #--------------------------------------------------
    # 2. change polar to cartesian coordinates
    #--------------------------------------------------
    f1c = convert_p2cart(f1p, p, t, freqs)
    # f1c = convert_p2cart(f1p)
    fs = f1c
    
    #--------------------------------------------------
    # 1. IFFT for f
    #--------------------------------------------------
    f0, ff = approx_cont_ft_SE2(fs, 0.0, 2*pi/size(fs,1), 0.0, 2*pi/size(fs,2) )
    @show ff
    # f0 = fft(fftshift(fs, (1,2)), (1,2)) # optional
    # f0 = fftshift(f0, (1,2))
    # f = permutedims(f0, (3,2,1))  # should be H W ntheta

    # f0: [H W m]
    # factor: nn psi
    f = zeros(ComplexF64, size(f0, 1), size(f0, 2), ntheta)

    for itheta=1:ntheta
        factor = exp.(-im * t[itheta] * nn) # size: nn 
        f[:,:,itheta] = sum(reshape(factor, 1, 1, :) .* f0, dims=3)
    end

    return f / (2.0*pi)
end





"""
# using ImageTransformations
# using NFFT
# using FINUFFT

# function map_coordinates(img, coord)
#     out = similar(coord[1])
#     itp = Interpolations.interpolate(img, BSpline(Linear()), OnCell())
#     etpf = extrapolate(itp, Line())   # gives 1 on the left edge and 7 on the right edge
#     # etpf = extrapolate(itp, Flat())   # gives 1 on the left edge and 7 on the right edge

#     for i=1:size(out,1), j=1:size(out,2)
#         # @show coord[1][i,j], coord[2][i,j]
#         out[i,j] = etpf[coord[1][i,j], coord[2][i,j]]
#     end
#     return out
# end

# function c2p_coords_spline(H, W, oversampling_factor = 1, npsi=nothing)
#     # p0 = H // 2
#     # q0 = W // 2
#     p0 = 0.49
#     q0 = 0.49

#     spline_order = 1
#     freq_max = 0.49
#     r_max = sqrt(freq_max^2 + freq_max^2)

#     # n_samples_r = oversampling_factor * (ceil(Int, r_max) + 1)
#     n_samples_r = H
#     if isnothing(npsi)
#         npsi = oversampling_factor * (ceil(Int, 2*pi*r_max))
#     end

#     r = LinRange(0, r_max, n_samples_r) #endpoint=true
#     t = LinRange(0, 2*pi, npsi+1)[1:end-1]

#     # meshgrid (ij)
#     rr = ones(length(t))' .* r
#     tt = t' .* ones(length(r))

#     X = rr .* cos.(tt)
#     Y = rr .* sin.(tt)

#     # change to image coordinates
#     I = X .+ abs(minimum(X))
#     J = Y .+ abs(minimum(Y))
#     return [I, J], r, t
# end

# function p2c_coords_spline(H, W, n_samples_r, npsi)
#     p0 = H // 2
#     q0 = W // 2
#     #---------------------------------------------------
#     # aaa
#     #---------------------------------------------------
#     i = 0:W-1
#     j = 0:H-1
#     x = i .- p0
#     y = j .- q0

#     xx = ones(length(y))' .* x
#     yy = y' .* ones(length(y))

#     R = sqrt.(xx.^2 + yy.^2)
#     T = atan.(yy, xx)

#     # if ~isnothing(n_samples_r)
#         r_max = sqrt(p0^2 + q0^2)

#         R .*= (n_samples_r - 1) / r_max
#         # TODO (to understand)
#         # The maximum angle in T is arbitrarily close to 2 pi,
#         # but this should end up 1 pixel past the last index npsi - 1, i.e. it should end up at npsi
#         # which is equal to index 0 since wraparound is used.
#         T .*= npsi / (2 * pi)
#     # end

#     # R .+= 1 # julia index starts from 1
#     # T .+= 1
#     return [R, T]
# end

# function wrap_coord(f, coord)
#     H, W, C = size(f)
#     f_wrap = zeros(typeof(f[1,1,1]), H+1, W+1, C)
#     f_wrap[1:end-1,1:end-1,:] = f
#     f_wrap[end,1:end-1,:] = f[1,:,:]
#     f_wrap[1:end-1,end,:] = f[:,1,:]
#     f_wrap[end,end,:] = f[1,1,:]

#     coord[1] = mod.(coord[1], H)
#     coord[2] = mod.(coord[2], W)

#     coord[1] .+= 1
#     coord[2] .+= 1

#     return f, coord
# end

# function convert_c2polar(f, npsi=nothing)
#     # warp(f, c2)
#     H, W, ntheta = size(f)
#     coord, rr, tt = c2p_coords_spline(H, W, 1, npsi)

#     f_wrap, coord = wrap_coord(f, coord)

#     r_sz = size(coord[1],1)
#     t_sz = size(coord[1],2)
#     f_polar_real = zeros(r_sz, t_sz, ntheta)
#     f_polar_im = zeros(r_sz, t_sz, ntheta)
#     for i = 1 : ntheta
#         f_polar_real[:,:,i] = map_coordinates(real(f_wrap[:,:,i]), coord)
#         f_polar_im[:,:,i] = map_coordinates(imag(f_wrap[:,:,i]), coord)
#     end
#     pp = rr
#     return f_polar_real .+ im .* f_polar_im, pp
# end




#--------------------------------------------------
# 2. change polar to cartesian coordinates
#--------------------------------------------------
convert_p2cart(f1p, freqs)
# f1c = convert_p2cart(f1p)
coord = p2c_coords_spline(H, W, size(fhat, 1), size(fhat, 2))
f1p_wrap, coord = wrap_coord(f1p, coord)

yy_sz, xx_sz = size(coord[1])
f_cart_real = zeros(yy_sz, xx_sz, ntheta)
f_cart_im = zeros(yy_sz, xx_sz, ntheta)

for i = 1 : ntheta
    f_cart_real[:,:,i] = map_coordinates(real(f1p_wrap[:,:,i]), coord)
    f_cart_im[:,:,i] = map_coordinates(imag(f1p_wrap[:,:,i]), coord)
end
f1c = f_cart_real .+ im .* f_cart_im

p0 = Int(H // 2); q0 = Int(W // 2);
y = mod.(-p0:-p0+H-1, H) .+ 1
x = mod.(-q0:-q0+W-1, W) .+ 1


# f1 = fftshift(f1c, (1,2)) # optional
# # f1 = f1c
# fs = fft(f1, (1,2)) # / (size(f1,1)*size(f1,2))
# f1 = conj(ifft(ifftshift(conj(f1c), (1,2)), (1,2))) # optional

fs = f1c

f1s, freqs = approx_cont_ft_SE2(conj(fs); use_ifft=true)
f1s = conj(f1s)

# if use_cont_ft
#     y0 = -H/2+0.5; x0 = -W/2+0.5
#     freqs = fftshift(fftfreq(H))

#     for j=1:W 
#         x0f = x0*freqs[j]
#         for i=1:H 
#             fs[i,j,:] .*= exp(im * (y0*freqs[i] + x0f) )
#         end
#     end
# end

# f1 = fft(fftshift(fs, (1,2)), (1,2)) # optional
# f1 = conj(ifft(ifftshift(conj(fs), (1,2)), (1,2))) # optional

# f = f1
# f = f1[y, x, :]

"""


