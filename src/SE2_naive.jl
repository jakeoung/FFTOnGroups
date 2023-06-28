using Bessels


function comp_bessel(order, x)
    if order == 1
        return Bessels.besselj1(x)
    elseif order == -1
        return -Bessels.besselj1(x)
    else
        return Bessels.besselj(order, x)
        # return bessel_j(float(order), x)
    end
end


# # build all the irreducible representations, 
# # each irrep is of shape (2L+1) x (2L+1)
# function build_irreps(G::SpecialEuclidean(2), L=4)
    
#     list_rep = []
#     # rot_order = Int(G.order // 2)
#     # khalf = Int(rot_order//2)

#     push!(list_rep, Rep(_build_irrep(G, 0, 0), 1))
#     push!(list_rep, Rep(_build_irrep(G, 1, 0), 1))
    
#     for k = 1 : G.max_freq -1
#         push!(list_rep, Rep(_build_irrep(G, 1, k), 2))
#     end

#     return list_rep
# end


# return irrep matrix of size (2L+1) x (2L+1)
function _build_irrep_SE2!(u, x, y, theta, p, nn, inv=false)
    # to polar coordinate
    a = sqrt(x^2 + y^2)
    phi = atan(y, x)
    # theta = atan(g.parts[2][2,1], g.parts[2][1,1])
    length_nn = length(nn)

    if inv == false
        # for n = -L : L
        for (in, n) in enumerate(nn)
            u[in, in] = exp(-im * n*theta) * besselj0(p*a)
            # u[n+L+1,n+L+1] = exp(-im * n*theta) * besselj0(p*a)
            
            for (idm, m) in enumerate(in+1 : length_nn)
                b = comp_bessel(n-m, p*a)
                nm = float(n-m)
                u[idm, in] = im ^nm * exp( -im * (n * theta - nm * phi)  ) * b
                # u[m+L+1,n+L+1] = im ^nm * exp( -im * (n * theta - nm * phi)  ) * b
            end
        end
    else
        for (in, n) in enumerate(nn)
            u[in, in] = exp(im * n*theta) * besselj0(p*a)

            for (idm, m) in enumerate(in+1 : length_nn)
                b = comp_bessel(m-n, p*a)
                nm = float(n-m)
                u[idm, in] = im ^nm * exp( im * (m * theta + nm * phi)  ) * b
                # u[m+L+1,n+L+1] = im ^nm * exp( im * (m * theta + nm * phi)  ) * b
            end
        end
    end

    # n=8
    # 1 -> 8
    # 2 -

    for (in, n) in enumerate(nn)
        for (idm, m) in enumerate(1 : in-1)
            u[idm, in] = conj(u[end-idm+1,end-in+1]) * -1^float(m-n)
    # for n = -L : L
        # for m = -L : n-1
            # u[m+L+1,n+L+1] = conj(u[-m+L+1,-n+L+1]) * -1^float(m-n)
        end
    end
end


"fourier transform"
function ft_SE2(f)
    # , xx, yy, thetas, pp=0:20, L=4
    H, W, ntheta = size(f)
    npsi = ntheta
    thetas = LinRange(0, 2*pi, npsi+1)[1:end-1]
    
    rr = -H//2 +0.5 : H//2 - 0.5
    cc = -W//2 +0.5 : W//2 - 0.5

    r_max = sqrt( (H//2)^2 + (W//2)^2)
    oversampling_factor = 1
    n_samples_r = oversampling_factor * (ceil(Int, r_max) + 1)

    # 1. generate irrepr u_nm(g',p)
    pp = LinRange(0, 0.5, n_samples_r)
    npp = length(pp)
    
    mm = Int.(fftshift(fftfreq(ntheta, ntheta)))
    nn = copy(mm)

    u = zeros(ComplexF32, length(mm), length(nn))
    fhat = zeros(ComplexF32, length(mm), length(nn), npp)

    for ipp = 1 : npp
        @info ipp
        # fhat(p)_ij = \sum_g f(g) U_ij (g^-1, p) dg
        for (ix, x) in enumerate(cc)
            for (iy, y) in enumerate(rr)
                for (itheta, theta) in enumerate(thetas)
                    _build_irrep_SE2!(u, x, y, theta, pp[ipp], nn, true)
                    fhat[:, :, ipp] += f[iy,ix,itheta] .* u
                end
            end
        end
    end
    
    return fhat, pp, mm, nn
end


"fourier transform by naive impl."
function ift_SE2(fhat, pp, H, W)
    rr = -H//2 +0.5 : H//2 - 0.5
    cc = -W//2 +0.5 : W//2 - 0.5

    # f(g) = \sum_p tr( fhat(p) U(g,p) )

    # 1. generate irrepr u_nm(g',p)
    npp = length(pp)
    
    ntheta = size(fhat, 1)
    f = zeros(typeof(fhat[1,1,1]), H, W, ntheta)
    U = zeros(typeof(fhat[1,1,1]), size(fhat, 1), size(fhat, 2))

    d = size(fhat, 1)
    thetas = LinRange(0, 2*pi, ntheta+1)[1:end-1]

    for (itheta, theta) in enumerate(thetas)
        @show itheta
        for (ix, x) in enumerate(cc) 
            for (iy, y) in enumerate(rr)
                # ipp = 1 -> pp[1] = 0
                for ipp = 2 : npp
                    _build_irrep_SE2!(U, x, y, theta, pp[ipp], nn, false)
                    f[iy, ix, itheta] += pp[ipp] * sum(transpose(fhat[:,:,ipp]) .* U)
                end
            end
        end
    end
    
    return f * d
end

# naive implementation
# See (11.1) by Chirikjian
function SE2_naive_conv(f1, f2, thetas)
    H, W, ntheta = size(f1)
    out = similar(f1)
    itp = Interpolations.interpolate(f2, BSpline(Linear()), OnCell())
    f2_interp = extrapolate(itp, 0.0)
    Rot = zeros(2,2)

    for (k,theta_outside) in enumerate(thetas)
        for i=1:H
            y = -( i -1 -H//2 + 0.5 ) # H=20. -9.5 ~ 9.5
            @show i
            Threads.@threads for j=1:W 
                x = j -1 -W//2 + 0.5

                value = 0.0*im
                for (kk,theta_inside) in enumerate(thetas)
                    # @show k
                    Rot .= [cos(theta_inside) sin(theta_inside); -sin(theta_inside) cos(theta_inside)]
                    for ii=1:H
                        yy = -( ii -1 -H//2 + 0.5 ) # H=20. -9.5 ~ 9.5
                        ydiff = y-yy
                        for jj=1:W 
                            xx = jj -1 -W//2 + 0.5
                
                            # f1(r, R) f2(R^−1 (a − r), R^−1 A)dr dR
                            # @show i,j (inverse matrix)
                            new_coord = Rot * [x-xx; ydiff]

                            kk_new = mod(-theta_inside + theta_outside, ntheta) + 1
                            value += f1[ii,jj,kk] * f2_interp[new_coord[1],new_coord[2],kk_new]

                            # if f2_interp[new_coord[1],new_coord[2],kk_new] > 0
                            #     @show new_coord, f2_interp[new_coord[1],new_coord[2],kk_new]
                            # end
                        end
                    end
                end
                out[i,j,k] = value / (H*W*ntheta)
            end
        end
    end
    return out
end

function test_ft_SE2()
    f_test = zeros(ComplexF64, 15,15,10)
    f_test[3:4, 3:11, :] .= 1.0
    fhat, pp, mm, nn = ft_SE2(f_test)
    f_test1 = ift_SE2(fhat, pp, size(f_test, 1), size(f_test, 2))
    mean(abs.(f_test .- f_test1))

    # fhat0_test, pp, mm, nn = fft_SE2(f_test)
    # f1 = ifft_SE2(fhat0_test, mm, size(f_test, 1), size(f_test,2))
    @show sum(abs.(f_test .- f1)) / length(f_test)
    save("../result/test_SE2.png", n01(abs.(f1[:,:,1])))
end

