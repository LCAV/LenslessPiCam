import numpy as np
import torch
from numpy.fft import fftshift, fft2, ifftshift, ifft2





def ft2(g, delta):
    """
    Compute 2D DFT.
    Inspired by Listing 2.5 of "Numerical Simulation of Optical Wave Propagation with Examples in
    MATLAB" (2010).
    Parameters
    ----------
    g : :py:class:`~numpy.ndarray`
        2D input samples.
    delta : float or list
        Sampling period along x-dimension and (optionally) y-dimension [m].
    """
    if isinstance(delta, float) or isinstance(delta, int):
        delta = [delta, delta]
    assert len(delta) == 2
    fact = delta[0] * delta[1]
    if torch.is_tensor(g):
        return torch.fft.fftshift(
            torch.fft.fft2(
                # TODO ifftshift of fftshift?
                torch.fft.fftshift(g * fact)
            )
        )
    else:
        res = fftshift(fft2(fftshift(g))) * fact
        if g.dtype == np.float32 or g.dtype == np.complex64:
            res = res.astype(np.complex64)
        return res



def ift2(G, delta_f):
    """
    Compute 2D IDFT.
    Inspired by Listing 2.6 from "Numerical Simulation of Optical Wave Propagation with Examples in
    MATLAB" (2010).
    Parameters
    ----------
    g : :py:class:`~numpy.ndarray`
        2D input samples.
    delta_f : float or list
        Frequency interval along x-dimension and (optionally) y-dimension [Hz].
    """
    if isinstance(delta_f, float) or isinstance(delta_f, int):
        delta_f = [delta_f, delta_f]
    assert len(delta_f) == 2
    fact = G.shape[0] * G.shape[1] * delta_f[0] * delta_f[1]
    # fact = 1   # TODO : huge difference when we don't apply factor
    if torch.is_tensor(G):
        # fact = pt.tensor([fact], dtype=G.dtype)
        return torch.fft.ifftshift(
            torch.fft.ifft2(
                # TODO ifftshift of fftshift?
                torch.fft.ifftshift(G * fact)
            )
        )
        # * G.shape[0] * G.shape[1] * delta_f[0] * delta_f[1]
    else:
        res = ifftshift(ifft2(ifftshift(G * fact)))
        if G.dtype == np.complex64:
            res = res.astype(np.complex64)
        return res



def sample_points(N, delta, shift=0, pytorch=False):
    """
    Return sample points in 2D.
    Parameters
    ----------
    N : int or list
        Number of sample points
    delta: int or float or list
        Sampling period along x-dimension and (optionally) y-dimension [m].
    shift : int or float or list
        Shift from optical axis
    """
    if isinstance(N, int):
        N = [N, N]
    assert len(N) == 2
    if isinstance(delta, float) or isinstance(delta, int):
        delta = [delta, delta]
    assert len(delta) == 2
    if isinstance(shift, float) or isinstance(shift, int):
        shift = [shift, shift]
    assert len(shift) == 2
    if pytorch:
        delta = torch.tensor(delta)
        shift = torch.tensor(shift)
        x = torch.arange(-N[1] / 2, N[1] / 2) * delta[1] + shift[1]
        x = torch.unsqueeze(x, 0)
        y = torch.arange(-N[0] / 2, N[0] / 2) * delta[0] + shift[0]
        y = torch.unsqueeze(y, 1)
    else:
        x = np.arange(-N[1] / 2, N[1] / 2)[np.newaxis, :] * delta[1] + shift[1]
        y = np.arange(-N[0] / 2, N[0] / 2)[:, np.newaxis] * delta[0] + shift[0]
    return x, y



def crop(u, shape, topleft=None, center_shift=None):
    """
    Crop center section of array or tensor (default). Otherwise set `topleft`.
    Parameters
    ----------
    u : array or tensor
        Data to crop.
    shape : tuple
        Target shape (Ny, Nx).
    Returns
    -------
    """
    Ny, Nx = shape
    if topleft is None:
        topleft = (int((u.shape[0] - Ny) / 2), int((u.shape[1] - Nx) / 2))
    if center_shift is not None:
        # subtract (positive) on second column to shift to the right
        topleft = (topleft[0] + center_shift[0], topleft[1] + center_shift[1])
    if torch.is_tensor(u):
        if u.dtype == torch.complex64 or u.dtype == torch.complex128:
            u_out_real = crop_torch(u.real, top=topleft[0], left=topleft[1], height=Ny, width=Nx)
            u_out_imag = crop_torch(u.imag, top=topleft[0], left=topleft[1], height=Ny, width=Nx)
            return torch.complex(u_out_real, u_out_imag)
        else:
            return crop_torch(u, top=topleft[0], left=topleft[1], height=Ny, width=Nx)
    else:
        return u[
            topleft[0] : topleft[0] + Ny,
            topleft[1] : topleft[1] + Nx,
        ]



def _get_dtypes(dtype, is_torch):
    if not is_torch:
        if dtype == np.float32 or dtype == np.complex64:
            return np.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return np.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)
    else:
        if dtype == np.float32 or dtype == np.complex64:
            return torch.complex64, np.complex64
        elif dtype == np.float64 or dtype == np.complex128:
            return torch.complex128, np.complex128
        elif dtype == torch.float32 or dtype == torch.complex64:
            return torch.complex64, np.complex64
        elif dtype == torch.float64 or dtype == torch.complex128:
            return torch.complex128, np.complex128
        else:
            raise ValueError("Unexpected dtype: ", dtype)



def zero_pad(u_in, pad=None):
    Ny, Nx = u_in.shape
    if pad is None:
        y_pad_edge = int(Ny // 2)
        x_pad_edge = int(Nx // 2)
    else:
        y_pad_edge, x_pad_edge = pad

    if torch.is_tensor(u_in):
        pad_width = (
            x_pad_edge + 1 if Nx % 2 else x_pad_edge,
            x_pad_edge,
            y_pad_edge + 1 if Ny % 2 else y_pad_edge,
            y_pad_edge,
        )
        return torch.nn.functional.pad(u_in, pad_width, mode="constant", value=0.0)
    else:
        pad_width = (
            (y_pad_edge + 1 if Ny % 2 else y_pad_edge, y_pad_edge),
            (x_pad_edge + 1 if Nx % 2 else x_pad_edge, x_pad_edge),
        )
        return np.pad(u_in, pad_width=pad_width, mode="constant", constant_values=0)







def fresnel_conv(u_in, wv, d1, dz, device=None, dtype=None, d2=None, pad=True):
    """
    Fresnel numerical computation (through convolution perspective) that gives
    control over output sampling but at a higher cost of two FFTs.
    Based off of Listing 6.5 of "Numerical Simulation of Optical Wave
    Propagation with Examples in MATLAB" (2010). Added zero-padding and support
    for PyTorch.
    NB: only works for square sampling, as non-square would result in different
    magnification factors.
    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [m].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [m].
    d2 : float or list
        Desired output sampling period for both x-dimension and y-dimension [m].
    dz : float
        Propagation distance [m].
    pad : bool
        Whether or not to zero-pad to linearize circular convolution. If the
        original signal has enough padding, this may not be necessary.
    device : "cpu" or "gpu"
        If using PyTorch, required. Device on which to perform computations.
    dtype : float32 or float64
        Data type to use.
    """
    if torch.is_tensor(u_in) or torch.is_tensor(dz):
        is_torch = True
    else:
        is_torch = False
    if is_torch:
        assert device is not None, "Set device for PyTorch"
        if torch.is_tensor(u_in):
            u_in = u_in.to(device)
        if torch.is_tensor(dz):
            dz = dz.to(device)
    assert isinstance(d1, float)
    if d2 is None:
        d2 = d1
    else:
        assert isinstance(d2, float)
    if dtype is None:
        dtype = u_in.dtype
    ctype, ctype_np = _get_dtypes(dtype, is_torch)

    if pad:
        N_orig = np.array(u_in.shape)
        u_in = zero_pad(u_in)
    N = np.array(u_in.shape)
    k = 2 * np.pi / wv

    # source coordinates
    x1, y1 = sample_points(N=N, delta=d1)
    r1sq = x1**2 + y1**2

    # source spatial frequencies
    df1 = 1 / (N * d1)
    fX, fY = sample_points(N=N, delta=df1)
    fsq = fX**2 + fY**2

    # scaling parameter
    m = d2 / d1

    # observation plane
    x2, y2 = sample_points(N=N, delta=d2)
    r2sq = x2**2 + y2**2

    # quadratic phase factors
    Q2 = np.exp(-1j * np.pi**2 * 2 * dz / m / k * fsq).astype(ctype_np)
    if is_torch:
        Q2 = torch.tensor(Q2, dtype=ctype).to(device)
    if m == 1:
        Q1 = 1
        Q3 = 1
    else:
        Q1 = np.exp(1j * k / 2 * (1 - m) / dz * r1sq).astype(ctype_np)
        Q3 = np.exp(1j * k / 2 * (m - 1) / (m * dz) * r2sq).astype(ctype_np)
        if is_torch:
            Q1 = torch.tensor(Q1, dtype=ctype).to(device)
            Q3 = torch.tensor(Q3, dtype=ctype).to(device)

    # propagated field
    u_out = Q3 * ift2(Q2 * ft2(Q1 * u_in / m, delta=d1), delta_f=df1)

    if pad:
        u_out = crop(u_out, shape=N_orig, topleft=(int(N_orig[0] // 2), int(N_orig[1] // 2)))

    return u_out, x2, y2