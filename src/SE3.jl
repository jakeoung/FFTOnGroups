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

function map_coordinates_3d(img, coord)
    out = similar(coord[1])
    itp = Interpolations.interpolate(img, BSpline(Linear()), OnCell())
    etpf = extrapolate(itp, Line())   # gives 1 on the left edge and 7 on the right edge
    # etpf = extrapolate(itp, Flat())   # gives 1 on the left edge and 7 on the right edge

    for i=1:size(out,1), j=1:size(out,2), k=1:size(out,3)
        # @show coord[1][i,j], coord[2][i,j]
        out[i,j,k] = etpf[coord[1][i,j], coord[2][i,j], coord[3][i,j]]
    end
    return out
end

function c2sp_coords_spline(H, W, Z, r_sample_factor = 1, npsi=nothing)
    Hhalf = H // 2
    Whalf = W // 2
    Zhalf = Z // 2

    # spline_order = 1
    r_max = sqrt(Hhalf^2 + Whalf^2 + Zhalf^2) 

    n_samples_r = r_sample_factor * (ceil(Int, r_max) + 1)
    if isnothing(npsi)
        npsi = r_sample_factor * (ceil(Int, 2*pi*r_max))
        # npsi = r_sample_factor * (ceil(Int, 2*pi*r_max))
    end

    r = LinRange(0, r_max, n_samples_r) 
    phi = LinRange(0, 2*pi, npsi+1)[1:end-1]
    t = LinRange(0, pi, npsi+1)[1:end-1]

    # meshgrid (ij)
    rr = reshape(ones(length(r)), :, 1, 1) .* reshape(ones(length(phi)), 1, :, 1) .* reshape(ones(length(t)), 1, 1, :)
    phis = reshape(ones(length(r)), :, 1, 1) .* reshape(phi, 1, :, 1) .* reshape(ones(length(t)), 1, 1, :)
    tt = reshape(ones(length(r)), :, 1, 1) .* reshape(ones(length(phi)), 1, :, 1) .* reshape(t, 1, 1, :)

    X1 = rr .* cos.(phis) .* sin.(tt)
    X2 = rr .* sin.(phis) .* sin.(tt)
    X3 = rr .* cos.(tt)

    I = X1 .+ Hhalf  # .+ 0.5 # we don't need, because after wrap, we will add 1
    J = X2 .+ Whalf  # .+ 0.5
    K = X3 .+ Zhalf

    return [I, J, K], r, phi, t
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

function extend3d_f(f, coord)
    H, W, Z, C = size(f)
    f_extended = zeros(typeof(f[1,1,1]), H+1, W+1, Z+1, C)
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

function convert_c2sp(f, npsi=nothing)
    # warp(f, c2)
    H, W, C, nangle = size(f)
    coord, rr, phis, tt = c2sp_coords_spline(H, W, C, 2, npsi)

    f_extended = f
    # f_extended, coord = extend3d_f(f, coord)

    r_sz, phi_sz, t_sz = size(coord[1])
    
    f_polar_real = zeros(r_sz, phi_sz, t_sz, nangle)
    f_polar_im = zeros(r_sz, phi_sz, t_sz, nangle)

    for i = 1 : nangle
        f_polar_real[:,:,:,i] = map_coordinates_3d(real(f_extended[:,:,:,i]), coord)
        f_polar_im[:,:,:,i] = map_coordinates_3d(imag(f_extended[:,:,:,i]), coord)
    end

    return f_polar_real .+ im .* f_polar_im, rr
end

# input: f [H, W, C, thetas]
# return: fhat(r,p,q)
# [m, n, p], m:theta, n:phi
function fft_SE3(f; use_cont_ft=true)
    npsi = size(f, 4)
    bnormalize = true

    #--------------------------------------------------
    # STEP 1. IFFT for f
    #--------------------------------------------------
    H, W, Z, ntheta = size(f)

    fs = ifftshift(f, (1,2,3))
    f1s = fftshift(fft(fs, (1,2,3)), (1,2,3)) 
    
    # See https://arxiv.org/pdf/1508.01282.pdf
    
    #--------------------------------------------------
    # 2. change cartesian to spherical coordinates
    #--------------------------------------------------
    # f1p [r, phi, theta]
    f1s, pp = convert_c2sp(f1s, npsi)

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

    varphi = LinRange(0.0, 2*pi, size(f2, 2)+1)[1:end-1]
    factor = exp.(-im .* reshape(varphi, 1,:,1) .* reshape(mm, 1,1,:))
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

    return fhat, pp, mm, nn
end

# inp: fhat_ of shape [m, n, p] (theta, psi, r)
function adjoint_fft_SE2(fhat_, H, W; mm=nothing, use_cont_ft=false)
    bnormalize = true
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
    f2f = conj(ifft(ifftshift(conj(fhat), (2)), (2)))
    (bnormalize) && (f2f /= (2*pi / size(f2f, 2)))
    
    psi = LinRange(0.0, 2*pi, size(f2f, 2)+1)[1:end-1]
    factor = exp.(im .* reshape(psi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2f .* factor

    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f1p = conj(ifft(ifftshift(conj(f2), (3)), (3)))
    (bnormalize) && (f1p /= (2*pi / ntheta))

    #--------------------------------------------------
    # 2. change polar to cartesian coordinates
    #--------------------------------------------------
    # f1c = convert_p2cart(f1p)
    # global coord, f1p_extended
    coord = p2c_coords_spline(H, W, size(fhat, 1), size(fhat, 2))
    f1p_extended, coord = extend_f(f1p, coord)
    # f1p_extended, coord = f1p, coord

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
    f = ifftshift(f, (1,2))
    # f = f1[y, x, :]

    # f1s = ifftshift(fs, (1,2))
    # f = conj(ifft(conj(f1s)))
    
    # f = ifft(f1)

    return f 
end

