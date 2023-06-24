using Manifolds
using LinearAlgebra
using ImageTransformations
using Interpolations
using CoordinateTransformations

using NFFT
using FINUFFT

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

    # R .+= 1 # julia index starts from 1
    # T .+= 1
    return [R, T]
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

function convert_c2polar(f, npsi=nothing)
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
    return f_polar_real .+ im .* f_polar_im, pp
end

# input: f [H, W, thetas]
# return: fhat(r,p,q)
# [m, n, p], m:theta, n:phi
function fft_SE2(f; use_cont_ft=true)
    npsi = size(f, 3)

    #--------------------------------------------------
    # STEP 1. IFFT for f
    #--------------------------------------------------
    H, W, ntheta = size(f)
    
    y = Int.(fftfreq(H,H)) .+ Int(H//2) .+ 1
    x = Int.(fftfreq(W,W)) .+ Int(W//2) .+ 1
    
    fs = f[y, x, :]
    f1s = ifftshift(ifft(fs, (1,2)), (1,2)) 

    if use_cont_ft
        y0 = -H/2+0.5; x0 = -W/2+0.5
        freqs = fftshift(fftfreq(H))
        # freqs = fftfreq(H)
        for i=1:H 
            for j =1:W 
                f1s[i,j,:] .*= exp(-im * (y0*freqs[i] + x0*freqs[j]))
            end
        end
    end

    # See https://arxiv.org/pdf/1508.01282.pdf
    
    #--------------------------------------------------
    # 2. change cartesian to polar coordinates
    #--------------------------------------------------
    # f1p [r, phi, theta]
    f1p, pp = convert_c2polar(f1s, npsi)

    mm = Int.(fftshift(fftfreq(size(f1p, 3), size(f1p, 3))))
    nn = Int.(fftshift(fftfreq(size(f1p, 2), size(f1p, 2))))
    
    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f2 = conj(fftshift(fft(conj(f1p), (3)), (3)))

    # m_min = -floor(Int, size(f2, 3) / 2)    # m_max = ceil(Int, size(f1p, 3) / 2) -1 # @show m_min, m_max, mm

    varphi = LinRange(0.0, 2*pi, size(f2, 2)+1)[1:end-1]
    # @show m_min, m_max, varphi, size(f1p)
    factor = exp.(-im .* reshape(varphi, 1,:,1) .* reshape(mm, 1,1,:))
    # @show size(factor), size(f2)
    f2 = f2 .* factor

    #--------------------------------------------------
    # 4. phi integration
    #--------------------------------------------------
    fhat = conj(fftshift( fft(conj(f2), (2)) , (2))) # / size(fhat, 2) # fhat(r,p,q)

    # [p, n, m] -> [m, n, p]
    fhat = permutedims(fhat, (3, 2, 1))

    return fhat, pp, mm, nn
end

# inp: fhat_ of shape [m, n, p] (theta, psi, r)
function ifft_SE2(fhat_, H, W; mm=nothing, use_cont_ft=true)
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
    
    psi = LinRange(0.0, 2*pi, size(f2f, 2)+1)[1:end-1]
    factor = exp.(im .* reshape(psi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2f .* factor

    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f1p = conj(ifft(ifftshift(conj(f2), (3)), (3)))
    # f1p = ifftshift(f1p, (3)) / size(f1p, 3)

    #--------------------------------------------------
    # 2. change polar to cartesian coordinates
    #--------------------------------------------------
    # f1c = convert_p2cart(f1p)
    # global coord, f1p_wrap
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

    f1 = fft(fftshift(fs, (1,2)), (1,2)) # optional
    # f1 = conj(ifft(ifftshift(conj(fs), (1,2)), (1,2))) # optional

    # f = f1
    f = f1[y, x, :]

    return f
end

function test_SE2()
    f_test = zeros(40,40,42)
    f_test[10:21, 11:30, :] .= 1.0
    fhat0_test, pp, mm, nn = fft_SE2(f_test)
    f1 = ifft_SE2(fhat0_test, size(f_test, 1), size(f_test,2))
    @show sum(abs.(f_test .- f1)) / length(f_test)
    save("../result/test_SE2.png", n01(abs.(f1[:,:,1])))
end

function test_conv()
    # f_test = zeros(40,40,10)
    # f_test[10:21, 11:13, :] .= 1.0
    f_test = zeros(40,40,3)
    f_test[10:15, 11:20, :] .= 1.0
    thetas = LinRange(0, 2*pi-(2*pi)/size(f_test,3), size(f_test,3));
    f2 = similar(f_test)
    f2 .= 0.1
    f_conv = SE2_conv(f_test, f2, thetas)
    # @show sum(abs.(f_test .- f1)) / length(f_test)
    save("../result/test_SE2_conv0.png", n01(abs.(f_test[:,:,2])))
    save("../result/test_SE2_conv.png", n01(abs.(f_conv[:,:,2])))
end

function imgpolarcoord(img::Array{Float64,2})
    # Adapted from: Javier Montoya (2020). Polar Coordinates Transform, MATLAB Central File Exchange
    # https://www.mathworks.com/matlabcentral/fileexchange/16094-polar-coordinates-transform
    
    (rows,cols) = size(img)
    cy, cx = rows ÷ 2, cols ÷ 2  
    radius = Int(minimum((cy,cx))) - 1
    Nϕ = 4*360
    Δϕ = 2pi/Nϕ
    pcimg = zeros(radius + 1, Nϕ)
    for  j = 1:Nϕ, i = 1:radius+1
            pcimg[i,j] = img[cy + round(Int,(i-1)*sin(Δϕ*(j-1))), cx + round(Int,(i-1)*cos(Δϕ*(j-1)))]
    end
    return pcimg
end


function fft_polar(f, use_nfft=false)
    H, W, ntheta = size(f)

    Hh = H // 2
    Wh = W // 2
    npsi = ntheta

    r_max = sqrt(Hh^2 + Wh^2)
    oversampling_factor = 1
    n_samples_r = oversampling_factor * (ceil(Int, r_max) + 1)
    if isnothing(npsi)
        npsi = oversampling_factor * (ceil(Int, 2*pi*r_max))
    end
    # @show n_samples_r

    # sqrt(0.5)
    r = LinRange(0, 0.5, n_samples_r) #endpoint=true
    t = LinRange(0, 2*pi, npsi+1)[1:end-1]

    coord = zeros(2, length(r)*length(t))
    cnt = 1

    pp = r
    phis = t

    @show "@@ debug"
    for tval in t 
        for rval in r

            x = rval * cos(tval)
            y = rval * sin(tval)
            coord[2,cnt] = -y
            coord[1,cnt] = x
            cnt += 1
        end
    end
    @show sum( (coord .< -0.5) .+ (coord .> 0.5))
    

    # 원점대칭 -> leads to e(-ipr) instead of e(ipr)
    if use_nfft == false 
        Af = mapslices(x -> nufft2d2(coord[2,:], coord[1,:], 1, 1e-9, x), f, dims=[1,2]) / (H*W)
        At = nothing

        # A = plan_nfft(coord, (H, W)) # , reltol=1e-9
        # At = adjoint(A)
        
    else 
        coord = clamp.(coord, -0.5, 0.5)
        A = plan_nfft(coord, (H, W)) # , reltol=1e-9
        Af = mapslices(x -> A*conj(x), f, dims=[1,2]) / (H*W)
        Af = conj(Af)
        At = adjoint(A)
    end

    # debug visualize Af
    # debug Af_img = reshape(Af, size(f))
    # scatter(coord[2,:], coord[1,:], markersize=Af[:,1,1])
    

    mm = Int.(fftshift(fftfreq(ntheta, ntheta)))
    nn = Int.(fftshift(fftfreq(length(t), length(t))))

    f1polar = reshape(Af, :, length(nn), ntheta)
    
    #---------------------------
    # 3. theta integration
    # f2 will be f2(r, phi, theta)
    #---------------------------
    f2 = conj(fftshift(fft(conj(f1polar), (3)), (3))) / ntheta
    # f2 = reshape(f1phat, :, length(nn), ntheta)

    varphi = LinRange(0.0, 2*pi, size(f2, 2)+1)[1:end-1]
    factor = exp.(-im .* reshape(varphi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2 .* factor

    #--------------------------------------------------
    # 4. phi integration
    #--------------------------------------------------
    # fhat = fhat(r,p,q)
    fhat = conj(fftshift( fft(conj(f2), (2)) , (2))) / size(f2, 2)

    # [p, n, m] -> [m, n, p]
    fhat = permutedims(fhat, (3, 2, 1))
    return fhat, pp, phis, mm, nn, At, coord
end

# inp: fhat of shape [m, n, p] (theta, phi, r)
function ifft_polar(fhat_, mm, nn, pp, phis, thetas, At, H, W)
    fhat = permutedims(fhat_, (3,2,1)) 
    ntheta = size(fhat, 3)
    # now fhat = fhat(r, phi, theta) = fhat(r, n, m)

    #--------------------------------------------------
    # 1. compute ftilde [r n n_outer]
    #--------------------------------------------------
    ftilde = similar(fhat)
    for (in, n) in enumerate(nn) # n
        phi = phis[in]
        for (ip, p) in enumerate(pp) # r
            for (in2, n2) in enumerate(nn) # n
                ftilde[ip,in,in2] = sum(fhat[ip, in2, :] .* exp.(im .* (n2 .- mm) * phi))
                ftilde[ip,in,in2] = ftilde[ip,in,in2] #* p
            end
        end
    end
    @show "ftilde done"

    #--------------------------------------------------
    # 1. INFFT
    #--------------------------------------------------
    # mapslices(x -> A*x, f, dims=[1,2])
    f1pvector = reshape(ftilde, :, length(nn))
    out0 = zeros(ComplexF64, H, W, length(nn))
    for n = 1 : length(nn)
        out0[:, :, n] = conj(At * conj(f1pvector[:,n]))
    end
    # out0 = out0 / ( length(pp) * length(phis) )
    @show "NFFT done"

    out = zeros(ComplexF64, H, W, length(thetas))
    for (imm, m) in enumerate(mm) # m, theta
        # temp_HxW = zeros(ComplexF64, H, W)
        for (in, n) in enumerate(nn)
            out[:,:,imm] += exp(-im .* n * thetas[imm]) * out0[:, :, in]
        end
    end
    out = out / (2.0*pi)
    return out
end

# # inp: fhat of shape [m, n, p] (theta, phi, r)
function ifft_polar0(fhat_, mm, At, H, W, coord)
    fhat = permutedims(fhat_, (3,2,1)) 
    ntheta = size(fhat, 3)
    # now fhat = fhat(r, phi, theta)

    #--------------------------------------------------
    # 4. phi integration
    #--------------------------------------------------
    f2f = conj(ifft(ifftshift(conj(fhat), (2)) *size(fhat, 2), (2)))

    psi = LinRange(0.0, 2*pi, size(f2f, 2)+1)[1:end-1]
    factor = exp.(im .* reshape(psi, 1,:,1) .* reshape(mm, 1,1,:))
    f2 = f2f .* factor

    #--------------------------------------------------
    # 3. integration on SO(2)
    #--------------------------------------------------
    f1p = conj(ifft(ifftshift(conj(f2 * size(f2, 3)), (3)), (3)))
    # f1p = ifftshift(f1p, (3)) / size(f1p, 3)
    
    # mapslices(x -> A*x, f, dims=[1,2])
    f1pvector = reshape(f1p, :, ntheta)
    
    out = zeros(ComplexF64, H, W, ntheta)

    if ~isnothing(At)
        for i = 1 : ntheta
            out[:, :, i] = At * conj(f1pvector[:,i])
        end 
        out = conj(out)
    else 
        # use nufft
        # polar -> cartesian
        # input non-uniform. output uniform
        for i = 1 : ntheta 
            nufft2d1!(coord[2,:], coord[1,:], f1pvector[:,i], -1, 1e-6, out[:,:,i])
        end
        # out = mapslices(x -> nufft2d1(coord[2,:], coord[1,:], x, -1, 1e-6, H, W), f1pvector, dims=[1])
        out = out / (H*W)
    end
    return out
end
