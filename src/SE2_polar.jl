using Manifolds
using LinearAlgebra
using ImageTransformations
using Interpolations
using CoordinateTransformations

using FFTW
using NFFT
using FINUFFT

n01(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))

include(srcdir("polar_dft.jl"))

# input: f [H, W, thetas]
# return: fhat(r,p,q)
# [m, n, p], m:theta, n:phi
function polar_fft_SE2(f; use_cont_ft=true)
    npsi = size(f, 3)
    bnormalize = false

    #--------------------------------------------------
    # STEP 1. IFFT for f
    #--------------------------------------------------
    H, W, ntheta = size(f)
    Hh = Int(H/2)
    # f = fftshift(f, (1,2))

    if mod(H, 2) == 0
        f_extended = zeros(typeof(f[1,1,1]), H+1, W+1, npsi)
        f_extended[1:end-1,1:end-1,:] = f
        f_extended[end,1:end-1,:] = f[end,:,:]
        f_extended[1:end-1,end,:] = f[:,end,:]
        f_extended[end,end,:] = f[end,end,:]
    else
        f_extended = f
    end
    f_extended = conj.(f_extended)

    f1p = zeros(typeof(f[1,1,1]), Int((H)/2)+1, npsi, npsi)
    # f1p = zeros(typeof(f[1,1,1]), H+1, npsi, npsi)

    # [nangles x radius]

    # DC component: Int(H/2)+1
    # psi: 0 ≤ psi < 180/M
    # psi: π ≤ psi+π < 2*π
    for ith=1:ntheta
        fp0_ang_r = compute_polar_dft(f_extended[:,:,ith], Int(npsi/2))
        # 1 2, 3, 4 5
        # vcat(fp0[Hh+1:end, :], fp0[Hh:-1:1, :]

        # what is the order of angle and radius?
        # angles: 


        # f1p[:,:,ith] = fp0_ang_r'
        
        # it is necessary to rearange
        fp0 = fp0_ang_r' # (r,psi)  
        v1 = vcat(fp0[Int(H/2)+1, :], fp0[Int(H/2)+1, :])'  # r = 0
        # @show size(v1) # 1 x 200
        v2 = hcat(fp0[Int(H/2)+2:end, :], fp0[Int(H/2):-1:1, :]) # r = 1~H/2
        # @show size(v2) # 25 x 200
        # v3 = hcat(fp0[Int(H/2)+2:end, :], fp0[Int(H/2):-1:1, :]) # r = 1~H/2

        f1p[:,:,ith] = vcat(v1, v2)
    end
    pp = Array(0.0:1.0:H/2)
    f1p = conj.(f1p)
    
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
function polar_ifft_SE2(fhat_, H_, W_, mm, nn; use_cont_ft=false)
    if mod(H_, 2) == 0
        H = H_+1
        W = W_+1
    else 
        H = H_ 
        W = W_
    end
    global f1p
    # if isnothing(mm)
    #     @assert mod(size(fhat_, 1), 2) == 0
    #     mm = fftshift(Int.(fftfreq(size(fhat_, 1), size(fhat_, 1))))
    #     nn = fftshift(Int.(fftfreq(size(fhat_, 2), size(fhat_, 2))))
    # end
    # t = LinRange(0, 2*pi, size(fhat_, 2)+1)[1:end-1]
    # ntheta = size(fhat_, 1)

    # # fhat_: (m,n,p)
    # pp_sz = size(fhat_, 3)
    # nn_sz = size(fhat_, 2)
    # ftilde = zeros(ComplexF64, pp_sz, length(t), length(t))

    # for in = 1 : nn_sz
    #     n = nn[in]
    #     nmt = reshape(n .- mm, :, 1, 1) .* reshape(t, 1, :, 1) # [ntheta x nn_sz x 1]
    #     exp_nmt = exp.(im * nmt)

    #     sum_1xNxP = sum(reshape(fhat_[in,:,:], :, 1, pp_sz) .* exp_nmt, dims=1)
    #     ftilde[:, :, in] =  permutedims(sum_1xNxP, (3,2,1))

    #     # for ip = 1 : pp_sz
    #     #     sum_fhat_1xnpsi = sum(reshape(fhat_[in,:,ip], :, 1) .* exp_nmt, dims=1)
    #     #     ftilde[ip, :, in] = sum_fhat_1xnpsi
    #     #     # for ipsi = 1 : nn_sz
    #     #     #     ftilde[ip, ipsi, in] = sum(fhat_[in,:,ip] .* exp_nmt
    #     #     #     # ftilde[ip, ipsi, in] = sum(fhat_[in,:,ip] .* exp.(im* (n .- mm) * t[ipsi] ))
    #     #     # end
    #     # end
    # end

    # f1p = ftilde

    bnormalize = false
    # if isnothing(mm)
    #     @assert mod(size(fhat_, 1), 2) == 0
    #     mm = collect(-size(fhat_, 1)//2 : size(fhat_, 1)//2-1)
    # end
    fhat = permutedims(fhat_, (3,2,1))
    ntheta = size(fhat, 3)
    # now (r, psi, theta)

    #--------------------------------------------------
    # 4. psi integration
    #--------------------------------------------------
    f2f = conj(ifft(ifftshift(conj(fhat), (2)), (2)))
    # (bnormalize) && (f2f /= (2*pi / size(f2f, 2)))
    (bnormalize) && (f2f *= size(f2f, 2) / sqrt(size(f2f, 2)) )
    
    psi = LinRange(0.0, 2*pi, size(f2f, 2)+1)[1:end-1]
    @show size(psi), size(mm)
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
    # r psi theta

    # polar_corners = Compute2DPolarCornersDFT2(H, ntheta)
    # Inverse2DPolarDFT

    f0 = zeros(ComplexF64, H, W, ntheta)
    Hh = Int((H-1)/2)
    for ith=1:ntheta
        f1temp = f1p[:,:,ith]
        @show size(f1temp)
        # r, theta
        # 1 2 3 4 5 6 7 (Hh = 3)
        # → 4 5 6 7 1 2 3
        # v1 = vcat(fp0[Int(H/2)+1, :], fp0[Int(H/2)+1, :])'  # r = 0
        # v2 = hcat(fp0[Int(H/2)+2:end, :], fp0[Int(H/2):-1:1, :]) # r = 1~H/2

        v1 = f1temp[end:-1:2, Int(end/2)+1:end]
        # v2 = f1temp[Hh+2:end, Int(end/2)+1:end]
        v2 = f1temp[1:end,   1:Int(end/2)]
        # v4 = f1temp[1:Hh+1,   1:Int(end/2)]
        vv = vcat(v1, v2)'
        @show size(vv) # size(vv) should be [H/2 x npsi/2]
        f0[:,:,ith] = Inverse2DPolarDFT(vv)
    end

    # coord = p2c_coords_spline(H, W, size(fhat_, 3), size(fhat_, 2))
    # # f1p_extended, coord = extend_f(f1p, coord)
    # f1p_extended, coord = f1p, coord

    # yy_sz, xx_sz = size(coord[1])
    # f_cart_real = zeros(yy_sz, xx_sz, ntheta)
    # f_cart_im = zeros(yy_sz, xx_sz, ntheta)

    # for i = 1 : ntheta
    #     f_cart_real[:,:,i] = map_coordinates(real(f1p_extended[:,:,i]), coord)
    #     f_cart_im[:,:,i] = map_coordinates(imag(f1p_extended[:,:,i]), coord)
    # end
    # f1c = f_cart_real .+ im .* f_cart_im
    # fs = f1c

    # f0 = fft(fftshift(fs, (1,2)), (1,2)) # optional
    # f0 = ifftshift(f0, (1,2)) 

    f = zeros(ComplexF64, size(f0, 1), size(f0, 2), ntheta)
    # factor = exp.(-im * reshape(t, 1,:) * reshape(nn, :, 1)) # size: nn x nn
    # sum(f0 * factor, 
    t = LinRange(0, 2*pi, ntheta+1)[1:end-1]

    for itheta=1:ntheta
        factor = exp.(-im * t[itheta] * nn) # size: nn 
        f[:,:,itheta] = sum(reshape(factor, 1, 1, :) .* f0, dims=3)
    end

    return f[1:end-1,1:end-1,:] #/ (2.0*pi)
end