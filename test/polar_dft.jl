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
    lineSpacing = -N/2:N/2;

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
