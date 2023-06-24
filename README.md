# FFTOnGroups

# Preliminary

Hello world

## Properties of the Fourier transform on R

### Compute the inverse Fourier transform via the forward Fourier transform

Let $*$ denotes the conjugation of the function.

$$\widehat{f^{*}}(w) = \hat{f}^*(-w)$$

This property can be useful to implement the inverse Fourier transform by the Fourier transform:

$$\widehat{f^{*}}^*(w) = \hat{f}(-w)$$

```
inverse_Fourier(f) =  conj(Fourier(conj(f)))
```

## DFT in Julia

$$\mathrm{DFT}(f)[k] = \sum_{j=0}^{N-1} e^{-ik x_j}f_j \quad 0 \le k \le N-1$$

$$\lbrace x_j \rbrace_{0 \le j \le N-1} = \left\lbrace \frac{2\pi j}{N} : \quad 0 \le j \le N-1 \right\rbrace$$


## Numerical approximation of continuous Fourier transform using DFT

$$F(k)=\int_{-\infty}^{\infty} d x e^{-i k x} f(x)$$
$$F(k) \approx \Delta x \sum_{m=0}^{N-1} e^{-i k x_{m}} f\left(x_{m}\right)$$
$\begin{aligned} F\left(k_{n}\right) & \approx \Delta x \sum_{m=0}^{N-1} \exp \left[-\frac{2 \pi i n\left(x_{0}+m \Delta x\right)}{N \Delta x}\right] f\left(x_{m}\right) \\ & =\Delta x \exp \left[-\frac{2 \pi i n x_{0}}{N \Delta x}\right] \sum_{m=0}^{N-1} e^{-i 2 \pi n m / N} f\left(x_{m}\right) \\ & =\Delta x \exp \left[-\frac{2 \pi i n x_{0}}{N \Delta x}\right] \operatorname{DFT}\left\{f\left(x_{m}\right)\right\}_{n}\end{aligned}$

$$k_{n} \equiv \frac{2 \pi n}{N \Delta x}$$

```python
w = np.fft.fftfreq(f.size)*2*np.pi/dt
```

## Reference

[1] https://www.dsprelated.com/showarticle/800.php shows some interesting ways to compute inverse FFT by forward FFT

[2] https://phys.libretexts.org/Bookshelves/Mathematical_Physics_and_Pedagogy/Computational_Physics_(Chong)/11%3A_Discrete_Fourier_Transforms/11.01%3A_Conversion_of_Continuous_Fourier_Transform_to_DFT