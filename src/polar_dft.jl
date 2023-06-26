using FFTW

function FrFTCenteredSingle( x, alpha, desiredIndex )
    #FRFTCENTEREDSINGLE computes the value using the given index k F^{alpha}(k)
    #using the data f(n)
    
    if length(size(x)) == 2
        sizeX, sizeY = size(x);
    else
        sizeX, sizeY = 1, length(x)
    end
    
    if (sizeX == 1)
        N = sizeY-1;  # N is even
        spacing = reshape(-N/2:1:N/2, 1, :);
    else
        N = sizeX-1;  # N is even
        spacing = [-N/2:1:N/2]';
    end
    
    scaleMultiples = exp.(-im*2*pi*spacing * alpha* desiredIndex/ (N+1));
    return sum(x[:] .* scaleMultiples[:])
end

function FrFT_Centered(x,alpha)

    #=====================================================================
    # Computes centered Fractional Fourier Transform according to the
    # definition given in the paper: "An exact and fast computation of Discrete Fourier
    # Transform for polar grid and spherical grid"
    #
    #           F^{\alpha}(k) = \sum\limits_{n=-N/2}^{N/2} f(n) e^{-j\frac{2\pi k\alpha n}{N+1}} ,
    #           -N/2 \leq k \leq N/2
    #
    # For alpha=1 we get the regular FFT, and for alpha= -1 we get the regular
    # IFFT up to a scale
    #
    # Synopsis: y=FRFT_Centered(x,alpha)   This is fully centered
    #
    # Inputs -    x - an N+1  - entry vector to be transformed
    #                   alpha - the arbitrary scaling factor
    #
    # Outputs-  y - the transformed result as an N+1-entries vector
    ====================================================================#
    
    if length(size(x)) == 2
        sizeX, sizeY = size(x);
    else
        sizeX, sizeY = 1, length(x)
    end
    
    if (sizeX == 1)
         N =sizeY-1;   # N is always assumed even
         n = [0:1:N; -N-1:1:-1]
         M = (0:1:N);
         k = (0:1:N);
         Z = zeros(N+1);
    else
        N =sizeX-1;   # N is always assumed even
        n = [0:1:N; -N-1:1:-1];
        M = (0:1:N)';
        k = (0:1:N)';
        Z = zeros(N+1,1);
    end
    M = M .- N/2;
    
    
    E_n = exp.(-im * pi * alpha * n.^2 / (N+1)) ;          # Sequence as defined in the paper
    PremultiplicationFactor = exp.(im * pi *M *N*alpha/(N+1));
    PostmultiplicationFactor = exp.(im * pi * alpha*N*k/(N+1));
    
    x_tilde=x .* reshape(PremultiplicationFactor, :, 1);
    
    if (sizeX == 1)
        x_tilde = [x_tilde; Z];
    else
        x_tilde = [x_tilde; Z];
    end
    
    x_tilde=x_tilde.*E_n;
    
    
    FirstFFT = fft(x_tilde);
    SecondFFT = fft(conj(E_n));
    interimY = ifft(FirstFFT.*SecondFFT);             # Computing Convolution here
    interimY = interimY.*E_n;
    y = PostmultiplicationFactor.*interimY[1:N+1];
    
    return y; 
end    

function VectorizedFrFT_Centered(x, alpha)

    # Check the dimensions of the input vector
    sizeX, _ = size(x)
    # if sizeX != 2
    #     error("The input vector must be N+1 in length")
    # end

    # Initialize the variables
    N = sizeX-1
    n = vcat(0:1:N, -N-1:1:-1)
    # n = range(-N/2, N/2)
    M = range(0, N)
    k = range(0, N)
    Z = zeros(size(x))
    M = M .- N/2

    n = reshape(n, :, 1)

    # Compute the sequence
    E_n = exp.(-1im * pi * alpha * n .^ 2 / (N+1))

    # Compute the pre-multiplication factor
    PremultiplicationFactor = exp.(1im * pi * M * N * alpha / (N+1))

    # Compute the post-multiplication factor
    PostmultiplicationFactor = exp.(1im * pi * alpha * N * k / (N+1))

    # Pad the input vector with zeros
    x_tilde = PremultiplicationFactor .* x
    x_tilde = vcat(x_tilde, Z)
    
    # Multiply the padded vector by the sequence
    x_tilde = x_tilde .* E_n

    # Compute the FFT of the padded vector
    FirstFFT = fft(x_tilde)

    # Compute the FFT of the sequence
    SecondFFT = fft(conj(E_n))

    # Compute the convolution of the two FFTs
    interimY = ifft(FirstFFT .* SecondFFT)

    # Multiply the convolution by the sequence
    interimY = interimY .* E_n

    # Compute the post-multiplication factor
    y = reshape(PostmultiplicationFactor, :, 1) .* interimY[1:N+1, :]

    return y

end


function compute_polar_dft(f, nangles)

    I = f
    sizeX, sizeY =  size(I);
    N = sizeX -1;      # N is always even
    M = nangles;    # M is also even

    # Three structures
    F_x_alpha_l = zeros(ComplexF64, size(I));   # X- scaled square grid
    F_y_alpha_l = zeros(ComplexF64, size(I));   # Y- scaled square grid
    PolarGrid = zeros(ComplexF64, M, N+1);     #  Polar grid: No of angles vs. Radial data

    dcValue = 0;
    lineData = zeros(ComplexF64, 1,N+1 );
    SymmlineData = zeros(ComplexF64, 1,N+1);
    lineSpacing = -N/2:N/2;   # radial line

    L = (M-2)/4;       # Number of levels
    hasFortyFiveDegrees = 0;

    if(rem(M-2,4) != 0)
        hasFortyFiveDegrees = 1;
        L = ceil(L);                   # use + 1
    end

    for l = 1 : L  # For each level 
        angle = l*180/M;
        alpha_l = cosd(angle);
        beta_l  = sind(angle);
        
        # X axis Scaling
        for x = 1:N+1
            row = f[x,:]
            FrFtSeq = FrFT_Centered(row, alpha_l);
            F_x_alpha_l[x,:] = FrFtSeq;
        end
        
        if (l == 1) # For first pass gather lines at 0 and 90
            NintyLine = F_x_alpha_l[:, Int(N/2)+1];          # Getting the column
            J = (0:1:N)';
            K = (0:1:N)';
            J = J .- N/2;
            PremultiplicationFactor = exp.(im*pi*J *N/(N+1));
            PostmultiplicationFactor = exp.(im*pi*N*K/(N+1));
            col = PostmultiplicationFactor[:] .* fft(NintyLine.*PremultiplicationFactor[:]);
            PolarGrid[Int(M/2)+1, :] = col; #fliplr ( conj(col'));
            line = PolarGrid[Int(M/2)+1,:];
            dcValue = line[Int(N/2)+1];
        end
        
        
        desiredIndexes = [-ones(1,Int(N/2))*l 0 ones(1,Int(N/2))*l];    
        # These are the desired locations where the Polar grid perfectly matches the indexes of the transformed 
        # butterfly indexes look at the figures in the paper
        
        for y =1 :N+1
            if (y != N/2+1) # Skip at zero computed seperately
                col = F_x_alpha_l[:,y];
                beta_factor = abs(lineSpacing[y])*beta_l /l;
                lineData[y] = FrFTCenteredSingle( col, beta_factor, desiredIndexes[y] );
                SymmlineData[y] = FrFTCenteredSingle( col, beta_factor, -desiredIndexes[y] );
            end
        end
        lineData[Int(N/2)+1] = dcValue;
        SymmlineData[Int(N/2)+1] = dcValue;
        PolarGrid[Int(1+l), :]= lineData;
        PolarGrid[Int(M+1-l), :]= reverse(SymmlineData, dims = 2)
        # fliplr( );
        
        if hasFortyFiveDegrees>0 && (angle == 45)   # If angle is 45 we have already computed it
            continue;
        end
        # Y axis Scaling
        for y = 1:N+1
            col = f[:,y];
            FrFtSeq = FrFT_Centered((col), alpha_l);
            F_y_alpha_l[:,y] = FrFtSeq ;
        end
        
        if (l == 1)
            ZeroLine = F_y_alpha_l[ Int(N/2)+1,:];
            J = (0:1:N);
            K = (0:1:N);
            J = J .- N/2;
            PremultiplicationFactor=exp.(im*pi*J *N/(N+1));
            PostmultiplicationFactor = exp.(im*pi*N*K/(N+1));
            PolarGrid[1,:] =  PostmultiplicationFactor .* fft(ZeroLine.*PremultiplicationFactor[:]);       # At 0 degrees
        end
        
        for x = 1:N+1
            if (x != N/2+1) # Skip at zero computed seperately
                row = F_y_alpha_l[x,:];
                beta_factor = abs(lineSpacing[x])*beta_l /l;
                lineData[x] = FrFTCenteredSingle( row, beta_factor, -desiredIndexes[x] );
                SymmlineData[x] = FrFTCenteredSingle(row, beta_factor, desiredIndexes[x] );
            end
        end
        lineData[Int(N/2)+1] = dcValue;
        SymmlineData[Int(N/2)+1] = dcValue;
        PolarGrid[Int(M/2+1+l), :]=  lineData ;
        PolarGrid[Int(M/2+1-l), :]=  SymmlineData;
    end
    return PolarGrid # [nangles x radius]
end
# F[m,n] theta_m = mΔθ,    -N/2 ≤ n ≤ N/2


function CornerPoints_2DDFT(sizeX, desiredPoint)

    N = sizeX -1;      # N is always even

    xIndex = desiredPoint[1]
    yIndex = desiredPoint[2]

    indexes = -N/2:N/2
    Map_x = exp.(-1im*2pi*xIndex*indexes/(N+1))
    Map_y = exp.(-1im*2pi*yIndex*indexes' /(N+1))

    # @show size(Map_y), size(I)

    lineData = dropdims(sum(I .* Map_y, dims=1), dims=1)
    ConjLineData = conj(lineData)

    @show size(lineData)

    FirstQuadPoint = sum(lineData .* Map_x)
    SecondQuadPoint = sum(ConjLineData .* Map_x)

    @show FirstQuadPoint, SecondQuadPoint
    ThirdQuadPoint = conj(SecondQuadPoint)
    FourthQuadPoint = conj(FirstQuadPoint)

    return vcat(FirstQuadPoint, SecondQuadPoint, ThirdQuadPoint, FourthQuadPoint)
end


function Compute2DPolarCornersDFT2(sizeX, noOfAngles)

    # I = inputImage
    # sizeX, _ =  size(I)
    N = sizeX -1;      # N is always even
    M = noOfAngles;    # M is also even

    deltaTheta = 180/M;                     # Angular sampling rate
    angles = deltaTheta:90-deltaTheta;      # Considering the first quadrant only  excluding the 0th and 90th for sure they dont and cant have any corner points

    FirstQuadPoints = []   
    PolarGridCornersWeights = []
    UniformGridSpacing = -N/2:N/2

    for angle in angles
        if ( angle <= 45 )                   # BH lines
            newGridSpacing = UniformGridSpacing ./ cos(angle*pi/180)         # BH   Since 0 < angle <= 45^o
        else
            newGridSpacing = UniformGridSpacing ./ sin(angle*pi/180)         # BV   Since 45 < angle < 90^o
        end
        halfSpacing = 1:1:newGridSpacing[end]
        halfPoints = halfSpacing[Int(N//2)+1:end]
        
        if (~isempty(halfPoints))
            for point in halfPoints
                push!(FirstQuadPoints, [point*cosd(angle),point*sind(angle)])
                push!(`PolarGridCornersWeights`, sqrt(point/2)/(N+1) )
            end
        end
    end
    FirstQuadPoints = hcat(FirstQuadPoints...)
    PolarGridCornersWeights = hcat(PolarGridCornersWeights...)
    PolarGridCornersWeights = PolarGridCornersWeights';  # Replicating for four corners

    ## Check the points if you have to
    # figure, 
    # hold on
    # scatter (FirstQuadPoints(:,1),FirstQuadPoints(:,2))
    # scatter (FirstQuadPoints(:,1),-FirstQuadPoints(:,2))
    # scatter (-FirstQuadPoints(:,1),FirstQuadPoints(:,2))
    # scatter (-FirstQuadPoints(:,1),-FirstQuadPoints(:,2))
    # hold off
    # axis equal; xlabel('x'); ylabel('y')

    @show size(FirstQuadPoints)
    C = size(FirstQuadPoints, 2)
    PolarGridCorners = zeros(ComplexF64, 4, C)

    for c = 1 : C
        point = FirstQuadPoints[:,c]
        xIndex = point[1]
        yIndex = point[2]
        PolarGridCorners[:,c] = CornerPoints_2DDFT(sizeX,[xIndex,yIndex])
    end
    return PolarGridCorners
end

function getBlockData2ComplementaryLines(lineData1, lineData2, beta_factor)

    # Tile the line data
    # tiledLineData = ones(sizeN_plus1, 1) * lineData1
    # tiledLineDataConj = ones(sizeN_plus1, 1) * lineData2
    tiledLineData = ones(length(lineData1), 1) .* reshape(lineData1, 1, :)
    tiledLineDataConj = ones(length(lineData1), 1) .* reshape(lineData2, 1, :)

    # Compute the variable-scale FFT
    frft_var_block1 = beta_factor * tiledLineData
    frft_var_block2 = conj(beta_factor) * tiledLineDataConj

    # Combine the two blocks
    block_data = frft_var_block1 + frft_var_block2

    return block_data

end


function Adjoint2DPolarDFT(Polar_Grid)

    # Number of rows and columns in the polar grid
    M, sizeN_plus1 = size(Polar_Grid)

    # Number of samples in the Cartesian grid
    N = sizeN_plus1 - 1

    # Number of levels in the FFT
    L = (M-2) / 4
    if (M-2) % 4 != 0
        L = ceil(L)
    end

    # Spacing between samples in the Cartesian grid
    lineSpacing = range(-N/2, N/2)
    gridSpacing = lineSpacing' * lineSpacing

    # Initialize the adjoint image
    ImageAdjoint = zeros(sizeN_plus1, sizeN_plus1)

    for l = 1:L
        l = Int(l)
        # Angle of the current level
        angle = l * 180 / M

        # Alpha and beta factors for the current level
        alpha_l = cosd(angle)
        beta_l = sind(angle)

        # Beta factor for the variable-scale FFT
        beta_factor = exp(2im * pi * beta_l * gridSpacing / sizeN_plus1)

        # Get the data from the two complementary lines
        line1 = Polar_Grid[1+l, :]
        # line2 = fliplr(Polar_Grid[M+1-l, :])
        line2 = reverse(Polar_Grid[M+1-l, :])

        # Compute the variable-scale FFT
        var_block = getBlockData2ComplementaryLines(line1, line2, beta_factor)

        # Compute the uniform-scale FFT
        frft_block = VectorizedFrFT_Centered(var_block', -alpha_l)

        # Add the contribution from the current level to the adjoint image
        ImageAdjoint = ImageAdjoint + frft_block

        # If the current angle is 45 degrees, there is no contribution from the Y-axis
        if angle == 45
            continue
        end

        # Get the data from the two complementary lines on the Y-axis
        line1 = Polar_Grid[Int(M/2)+1-l, :]
        line2 = Polar_Grid[Int(M/2)+1+l, :]

        # Compute the variable-scale FFT
        var_block = getBlockData2ComplementaryLines(line1, line2, beta_factor)

        # Compute the uniform-scale FFT
        frft_block = VectorizedFrFT_Centered(var_block', -alpha_l)

        # Add the contribution from the current level to the adjoint image
        ImageAdjoint = ImageAdjoint + frft_block'

        ## computing for zero and ninety degrees seperately

        if (l == 1)
            ZeroLine = reshape(Polar_Grid[1,:], 1,:)
            NinetyLine = reshape(Polar_Grid[Int(M/2)+1,:], 1,:)
            
            FrFTVarBlock1 =  ones(sizeN_plus1,1) .* NinetyLine;        # beta factors is all ones
            FrFTVarBlock2 =  ones(sizeN_plus1,1) .* ZeroLine;    # beta factors is all ones
            
            # computing uniform scale FrFT second
            FrFTUniformBlock_x = VectorizedFrFT_Centered(FrFTVarBlock2' , -1);   # Zero
            FrFTUniformBlock_y = VectorizedFrFT_Centered(FrFTVarBlock1' , -1);
            
            ImageAdjoint = ImageAdjoint + FrFTUniformBlock_x + FrFTUniformBlock_y';   # collecting contribution from every level
        end

    end

    # Return the adjoint image
    return ImageAdjoint

end


function Inverse2DPolarDFT(Polar_Grid)

    accuracy=1e-5;
    
    M, sizeN_plus1 = size(Polar_Grid);
    N = sizeN_plus1 -1;
    ImageFinal = zeros(N+1,N+1);
    
    fullSize  = sizeN_plus1;
    W         = sqrt.(abs.(-N/2:N/2)/2)/fullSize;    # Fourier based preconditioner from Amir's paper
    
    W[Int(N/2)+1]  =  sqrt(1/8)/fullSize;

    W        = ones(M,1) .* reshape(W, 1, :)
      
    ## Simple Gradient Descent
    Delta=1;
    count=0;
    maxIterations = 10;
    
    while Delta>accuracy && count < maxIterations
        
        ## No preconditioner 
        Err = W .* (compute_polar_dft( ImageFinal,  M ) - Polar_Grid);

        D = Adjoint2DPolarDFT(W .* Err)';

        Delta=norm(D);
        println("At iteration $count, the difference matrix norm is $Delta")
        
        mu = 1 / sizeN_plus1;
        mu *= 0.3;
        Temp = ImageFinal-mu*D; 
        ImageFinal = Temp;
        count=count+1;  

        # if (count >= 1 && count <= 3)
        #     figure, imshow(real(ImageFinal), [])
        #     str = strcat('Retrieval at iteration no.' , num2str(count));
        #     title (str);
        # end
    end
    
    println("Number of required iterations is $count")
    
    # figure, imshow(real(ImageFinal), [])
    # % figure, imagesc(imadjust(real(ImageFinal)))
    # str = strcat('Retrieval at iteration no.' , num2str(count));
    # title (str );
     
    return ImageFinal
end 
    

# function Compute2DPolarDFT(inputImage, noOfAngles)

#     I = inputImage
#     [sizeX, sizeY] = size(I)
#     N = sizeX - 1      # N is always even
#     M = noOfAngles     # M is also even

#     PolarGrid = zeros(M, N + 1)
#     lineData = zeros(1, N + 1)
#     SymmlineData = zeros(1, N + 1)

#     dcValue = 0

#     lineSpacing = -N/2:N/2

#     L = (M - 2) / 4
#     hasFortyFiveDegrees = 0

#     if (rem(M - 2, 4) != 0)
#         hasFortyFiveDegrees = 1
#         L = ceil(L)
#     end

#     for l = 1:L
#         angle = l * 180 / M
#         alpha_l = cos(angle)
#         beta_l = sin(angle)

#         F_x_alpha_l = VectorizedFrFT_Centered(inputImage', alpha_l)
#         F_x_alpha_l = F_x_alpha_l'

#         if (l == 1)
#             NintyLine = F_x_alpha_l[:, N/2 + 1]
#             J = (0:1:N)'
#             K = (0:1:N)'
#             J = J - N/2
#             premultiplication_factor = exp(1im * pi * J * N / (N + 1))
#             postmultiplication_factor = exp(1im * pi * N * K / (N + 1))
#             col = premultiplication_factor * fft(postmultiplication_factor * NintyLine)
#             PolarGrid[M/2 + 1, :] = col
#             fliplr(conj(col'))
#             line = PolarGrid[M/2 + 1, :]
#             dcValue = line[N/2 + 1]
#         end

#         desiredIndexes = [-ones(1, N/2) * l, 0, ones(1, N/2) * l]
#         for y = 1:N + 1
#             if (y != N/2 + 1)
#                 col = F_x_alpha_l[:, y]
#                 beta_factor = abs(lineSpacing(y)) * beta_l / l
#                 lineData[y] = VectorizedFrFTCenteredSingle(col, beta_factor, desiredIndexes[y])
#                 SymmlineData[y] = VectorizedFrFTCenteredSingle(col, beta_factor, -desiredIndexes[y])
#             end
#         end

#         lineData[N/2 + 1] = dcValue
#         SymmlineData[N/2 + 1] = dcValue
#         PolarGrid[1 + l, :] = lineData
#         PolarGrid[M + 1 - l, :] = fliplr(SymmlineData)

#         if (hasFortyFiveDegrees && angle == 45)
#             continue
#         end

#         F_y_alpha_l = VectorizedFrFT_Centered(inputImage, alpha_l)

#         if (l == 1)
#             ZeroLine = F_y_alpha_l[N/2 + 1, :]
#             J = (0:1:N)
#             K = (0:1:N)
#             J = J - N/2
#             premultiplication_factor = exp(1im * pi * J * N / (N + 1))
#             postmultiplication_factor = exp(1im * pi * N * K / (N + 1))
#             PolarGrid[1, :] = premultiplication_factor * fft(postmultiplication_factor * ZeroLine)
#         end

#         for x = 1:N + 1
#             if (x != N/2 + 1)
#                 row = F_y_alpha_l[x, :]
#                 beta_factor = abs(lineSpacing(x)) * beta_l / l
#                 lineData[x] = VectorizedFrFTCenteredSingle(row, beta_factor, -desiredIndexes[x])
#                 Symmline
