using Manifolds
using LinearAlgebra
using ImageTransformations
using Interpolations
using CoordinateTransformations

using FFTW
using NFFT
using FINUFFT

n01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))

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

function map_coordinates(img, coord)
    out = similar(coord[1])
    # itp = Interpolations.interpolate(img, BSpline(Linear()), OnCell())
    itp = Interpolations.interpolate(img, BSpline(Quadratic(Reflect(OnCell()))), OnCell())
    etpf = extrapolate(itp, Line()) 
    # etpf = extrapolate(itp, Flat())   # gives 1 on the left edge and 7 on the right edge

    for i=1:size(out,1), j=1:size(out,2)
        # @show coord[1][i,j], coord[2][i,j]
        out[i,j] = etpf[coord[1][i,j], coord[2][i,j]]
    end
    return out
end

function c2p_coords_spline(H, W, r_sample_factor, npsi=nothing)
    Hhalf = H // 2 
    Whalf = W // 2 

    spline_order = 1
    r_max = sqrt(Hhalf^2 + Whalf^2) 

    n_samples_r = r_sample_factor * (ceil(Int, r_max) + 1)
    if isnothing(npsi)
        npsi = r_sample_factor * (ceil(Int, 2*pi*r_max))
    end

    r = LinRange(0.00001, r_max, n_samples_r) 
    t = LinRange(0, 2*pi, npsi+1)[1:end-1]

    # meshgrid (ij)
    rr = ones(length(t))' .* r
    tt = t' .* ones(length(r))

    X = rr .* cos.(tt)
    Y = rr .* sin.(tt)

    # I = X .+ Hhalf
    # J = Y .+ Whalf
    I = X .+ Hhalf  # .+ 0.5 # we don't need, because after wrap, we will add 1
    J = Y .+ Whalf  # .+ 0.5

    return [I, J], r, t
end

function p2c_coords_spline(H, W, n_samples_r, npsi)
    Hhalf = H // 2
    Whalf = W // 2
    #---------------------------------------------------
    # aaa
    #---------------------------------------------------
    i = 0:W-1
    j = 0:H-1
    x = i .- Hhalf
    y = j .- Whalf

    xx = ones(length(y))' .* x
    yy = y' .* ones(length(y))

    R = sqrt.(xx.^2 + yy.^2)
    T = atan.(yy, xx)

    r_max = sqrt(Hhalf^2 + Whalf^2)

    R *= (n_samples_r - 1) / r_max
    # R *= (n_samples_r) / r_max
    T *= npsi / (2 * pi)

    # R .+= 1 # julia index starts from 1
    # T .+= 1
    return [R, T]
end

function extend_f(f, coord)
    H, W, C = size(f)
    f_extended = zeros(typeof(f[1,1,1]), H+1, W+1, C)
    f_extended[1:end-1,1:end-1,:] = f
    f_extended[end,1:end-1,:] = f[1,:,:]
    f_extended[1:end-1,end,:] = f[:,1,:]
    f_extended[end,end,:] = f[1,1,:]

    coord[1] = mod.(coord[1], H)
    coord[2] = mod.(coord[2], W)

    coord[1] .+= 1
    coord[2] .+= 1

    return f_extended, coord
end

function convert_c2polar(f, npsi=nothing, r_sample_factor=8, extend=true)
    # warp(f, c2)
    H, W, ntheta = size(f)
    coord, rr, tt = c2p_coords_spline(H, W, r_sample_factor, npsi)

    if extend
        f_extended, coord = extend_f(f, coord)
    else
        f_extended, coord = f, coord
    end

    r_sz = size(coord[1],1)
    t_sz = size(coord[1],2)
    f_polar_real = zeros(r_sz, t_sz, ntheta)
    f_polar_im = zeros(r_sz, t_sz, ntheta)

    for i = 1 : ntheta
        f_polar_real[:,:,i] = map_coordinates(real(f_extended[:,:,i]), coord)
        f_polar_im[:,:,i] = map_coordinates(imag(f_extended[:,:,i]), coord)
    end

    return f_polar_real .+ im .* f_polar_im, rr
end

@doc raw"""
    fft_SE2(f; use_cont_ft=false) -> fhat

Returns the Fourier transform fhat [m, n, p] m:theta, n:phi

# Arguments
- f [H, W, thetas]

# References
- https://docs.julialang.org/en/v1/manual/documentation/index.html
- the book by Chirikjian 
"""
function fft_SE2(f; r_sample_factor=16, use_cont_ft=false, bextend=true)
    npsi = size(f, 3)
    bnormalize = false

    #--------------------------------------------------
    # STEP 1. IFFT for f
    #--------------------------------------------------
    H, W, ntheta = size(f)

    # See https://dsp.stackexchange.com/questions/66716/why-do-we-have-to-rearrange-a-vector-and-shift-the-zero-point-to-the-first-index
    fs = ifftshift(f, (1,2))

    if use_cont_ft
        f1s, freqs = approx_cont_ft_SE2(conj(fs); use_ifft=false)
        f1s = conj(f1s) / (H*W)
    else
        f1s = fftshift(ifft(fs, (1,2)), (1,2))
        # f1s *= H*W / sqrt(H*W)
        # f1s *= (2*pi/H)*(2*pi/W) / sqrt(H*W) * H*W
    end

    # See https://arxiv.org/pdf/1508.01282.pdf
    
    #--------------------------------------------------
    # 2. change cartesian to polar coordinates
    #--------------------------------------------------
    # f1p [r, phi, theta]
    f1p, pp = convert_c2polar(f1s, npsi, r_sample_factor, bextend)

    mm = Int.(fftshift(fftfreq(size(f1p, 3), size(f1p, 3))))
    nn = Int.(fftshift(fftfreq(size(f1p, 2), size(f1p, 2))))
    
    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f2 = conj(fftshift(fft(conj(f1p), (3)), (3))) 
    if bnormalize
        # f2 *= ( 2*pi / ntheta ) # / sqrt(ntheta)
        f2 /= sqrt(ntheta)
    end

    varphi = LinRange(0.0, 2*pi, size(f2, 2)+1)[1:end-1]
    factor = exp.(-im .* reshape(varphi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2 .* factor

    #--------------------------------------------------
    # 4. psi integration
    #--------------------------------------------------
    fhat = conj(fftshift( fft(conj(f2), (2)) , (2))) * 2*pi / ntheta # / size(fhat, 2) # fhat(r,p,q)
    if bnormalize
        # fhat *= (2*pi / size(f2, 2)) # / sqrt(ntheta)
        fhat /= sqrt(ntheta)
    end

    # [p, n, m] -> [m, n, p]
    fhat = permutedims(fhat, (3, 2, 1))

    return fhat, pp, mm, nn
end

# inp: fhat_ of shape [m, n, p] (theta, psi, p)
function adjoint_fft_SE2(fhat_, H, W; mm=nothing, use_cont_ft=false, bextend=true)
    bnormalize = false
    if isnothing(mm)
        @assert mod(size(fhat_, 1), 2) == 0
        mm = collect(-size(fhat_, 1)//2 : size(fhat_, 1)//2-1)
    end
    fhat = permutedims(fhat_, (3,2,1))
    ntheta = size(fhat, 3)
    # now (r, psi, theta)

    #--------------------------------------------------
    # 4. psi integration
    #--------------------------------------------------
    f2f = conj(ifft(ifftshift(conj(fhat), (2)), (2))) * ntheta / (2*pi)
    # (bnormalize) && (f2f /= (2*pi / size(f2f, 2)))
    (bnormalize) && (f2f *= size(f2f, 2) / sqrt(size(f2f, 2)) )
    
    psi = LinRange(0.0, 2*pi, size(f2f, 2)+1)[1:end-1]
    factor = exp.(im .* reshape(psi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2f .* factor

    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f1p = conj(ifft(ifftshift(conj(f2), (3)), (3)))
    # (bnormalize) && (f1p /= (2*pi / ntheta))
    (bnormalize) && (f1p *= ntheta / sqrt(ntheta)  )

    #--------------------------------------------------
    # 2. change polar to cartesian coordinates
    #--------------------------------------------------
    # f1c = convert_p2cart(f1p)
    # global coord, f1p_extended
    coord = p2c_coords_spline(H, W, size(fhat, 1), size(fhat, 2))
    if bextend
        f1p_extended, coord = extend_f(f1p, coord)
    else
        f1p_extended, coord = f1p, coord
    end

    yy_sz, xx_sz = size(coord[1])
    f_cart_real = zeros(yy_sz, xx_sz, ntheta)
    f_cart_im = zeros(yy_sz, xx_sz, ntheta)

    for i = 1 : ntheta
        f_cart_real[:,:,i] = map_coordinates(real(f1p_extended[:,:,i]), coord)
        f_cart_im[:,:,i] = map_coordinates(imag(f1p_extended[:,:,i]), coord)
    end
    f1c = f_cart_real .+ im .* f_cart_im
    fs = f1c

    if use_cont_ft
        y0 = -H/2+0.5; x0 = -W/2+0.5
        freqs = fftshift(fftfreq(H))

        for j=1:W 
            x0f = x0*freqs[j]
            for i=1:H 
                fs[i,j,:] .*= exp(im * (y0*freqs[i] + x0f) )
            end
        end
    end

    f = fft(fftshift(fs, (1,2)), (1,2)) # optional
    f = ifftshift(f, (1,2)) # / sqrt(H*W)

    return f 
end

# inp: fhat_ of shape [m, n, p] (theta, psi, r)
function ifft_SE2(fhat_, H, W; mm=nothing, use_cont_ft=false, bextend=true)
    if isnothing(mm)
        @assert mod(size(fhat_, 1), 2) == 0
        mm = fftshift(Int.(fftfreq(size(fhat_, 1), size(fhat_, 1))))
        nn = fftshift(Int.(fftfreq(size(fhat_, 2), size(fhat_, 2))))
    end
    t = LinRange(0, 2*pi, size(fhat_, 2)+1)[1:end-1]
    ntheta = size(fhat_, 1)

    # fhat_: (m,n,p)
    pp_sz = size(fhat_, 3)
    nn_sz = size(fhat_, 2)
    ftilde = zeros(ComplexF64, pp_sz, length(t), length(t))

    for in = 1 : nn_sz
        n = nn[in]
        nmt = reshape(n .- mm, :, 1, 1) .* reshape(t, 1, :, 1) # [ntheta x nn_sz x 1]
        exp_nmt = exp.(im * permutedims(nmt, (1, 2, 3)))

        sum_1xNxP = sum(reshape(fhat_[in,:,:], :, 1, pp_sz) .* exp_nmt, dims=1)
        ftilde[:, :, in] =  permutedims(sum_1xNxP, (3,2,1))

        # for ip = 1 : pp_sz
        #     sum_fhat_1xnpsi = sum(reshape(fhat_[in,:,ip], :, 1) .* exp_nmt, dims=1)
        #     ftilde[ip, :, in] = sum_fhat_1xnpsi
        #     # for ipsi = 1 : nn_sz
        #     #     ftilde[ip, ipsi, in] = sum(fhat_[in,:,ip] .* exp_nmt
        #     #     # ftilde[ip, ipsi, in] = sum(fhat_[in,:,ip] .* exp.(im* (n .- mm) * t[ipsi] ))
        #     # end
        # end
    end

    f1p = ftilde

    #--------------------------------------------------
    # 2. change polar to cartesian coordinates
    #--------------------------------------------------
    # f1c = convert_p2cart(f1p)
    # r psi theta
    coord = p2c_coords_spline(H, W, size(fhat_, 3), size(fhat_, 2))
    if bextend
        f1p_extended, coord = extend_f(f1p, coord)
    else
        f1p_extended, coord = f1p, coord
    end

    # f1p_extended /= sqrt(pp_sz)

    yy_sz, xx_sz = size(coord[1])
    f_cart_real = zeros(yy_sz, xx_sz, ntheta)
    f_cart_im = zeros(yy_sz, xx_sz, ntheta)

    for i = 1 : ntheta
        f_cart_real[:,:,i] = map_coordinates(real(f1p_extended[:,:,i]), coord)
        f_cart_im[:,:,i] = map_coordinates(imag(f1p_extended[:,:,i]), coord)
    end
    f1c = f_cart_real .+ im .* f_cart_im
    fs = f1c

    if use_cont_ft 
        # to be modified
        f0, _ = approx_cont_ft_SE2(fftshift(fs, (1,2)), 0.0, 2*pi/size(fs,1), 0.0, 2*pi/size(fs,2))
        f0 = ifftshift(f0, (1,2))
    else
        f0 = fft(fftshift(fs, (1,2)), (1,2)) # optional
        f0 = ifftshift(f0, (1,2)) 
        f0 *= 1.0 / (H*W)
    end

    f = zeros(ComplexF64, size(f0, 1), size(f0, 2), ntheta)
    # factor = exp.(-im * reshape(t, 1,:) * reshape(nn, :, 1)) # size: nn x nn
    # sum(f0 * factor, 
    for itheta=1:ntheta
        factor = exp.(-im * t[itheta] * nn) # size: nn 
        f[:,:,itheta] = sum(reshape(factor, 1, 1, :) .* f0, dims=3)
    end
    
    return f / (2.0*pi) # / 10
end


