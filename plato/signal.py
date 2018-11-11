#!/usr/bin/env python
"""
    Signal processing tools

    function:
       nfreqreal
       numofensembles
       normalize
       psd
       xcoh2
       xphase
       csd
       psd2
       running
       bicoherence_simple
       abicoh2
       xbicoh2
       summedbicoh2

    Status
    ------
    Version 0.9.2

    Authour
    -------
    Shigeru Inagaki                                       
    Research Institute for Applied Mechanics 
    inagaki@riam.kyushu-u.ac.jp  
    
    Revision History
    ----------------
    [30-March-2015]   Creation (v0.8)
    [18-October-2016] Function "normalize" added (v0.8.1)
                      psd(sigs, t, nfft=256, dt=None, noverlap=None, nensemble=1, istart=0, tstart=None, window='hanning', detrend=normalize)
    [17-April-2017]   Bugs in dt-evaluation fixed (v0.8.2)
    [18-April-2017]   Function "psd2d" added (v0.8.3)
    [05-May-2017]     Function "bicoherence_simple, abicoh2, xbicoh2, summedbicoh2" added (v0.9.0)
    [17-May-2017]     Function xcoh2 is fixed (Thanks Kin!)
    [19-May-2017]     comb key word has benn added on xbicoh2 (v0.9.1)
    [01-Jun-2017]     axisf1 option added in summedbicoh2 (v0.9.2)

    Copyright
    ---------
    2017 Shigeru Inagaki (inagaki@riam.kyushu-u.ac.jp)
    Released under the MIT, BSD, and GPL Licenses.

"""

import warnings
import numpy as np
import scipy.fftpack as fft
import scipy.signal as dsp
import scipy.linalg as la

    #周波数を求める?
def nfreqreal(nfft): 
    if nfft % 2 == 0 :
       nfreq = nfft//2 + 1
    else:
       nfreq = (nfft+1)//2
    return nfreq
    
    #パワースペクトル密度を求める
def psd(sigs, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hanning', detrend='constant'):
    """
    Estimate auto power spectral density using Welch's method.
    Welch's method [1]_ computes an estimate of the power spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.
    Parameters
    ----------
    sigs : array_like
        Time series of measurement values
    t : 1D-array
        Time for sigs
    x : 1D-array
        Time series of reference value
    dt : float, optional
        Sampling time of the `x` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nfft / 2``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'Hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxx : ndarray
        Power spectral density or power spectrum of sigs.
    See Also
    --------
    psd:power spectrum density
    running: running FFT
    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements.  For the default 'hanning' window an
    overlap of 50% is a reasonable trade off between accurately estimating
    the signal power, while not over counting any of the data.  Narrower
    windows may require a larger overlap.
    If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
    .. versionadded:: 0.12.0
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
   
    if dt is None: 
        dt = (t[-1] - t[0]) / (len(t)-1) #時間設定

    if tstart is not None:
        indx = np.where(t >= tstart)[0] #スタート時間設定
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2 #計算時にかぶる配列の長さを設定
        
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart #終わりの時間設定　アンサンブル分だけの和を取る

    if iend > len(t) :
       print("Index is out of bounds for axis 0")
       return np.empty(t.shape), np.empty(aigs.shape) #最終配列の番号が問題の時の設定

    f, Pxx = dsp.welch(sigs[istart:iend, ...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None, detrend=detrend, return_onesided=True, scaling='density', axis=0)
    #ウェルチ法を用いてパワースペクトルを求める
    print(sigs[istart:iend, ...])
    #第一パラメータだけ表示?
    return f, Pxx


    #アンサンブル数の設定
def numofensembles(t, nfft=256, noverlap=None, istart=0, iend=-1, tstart=None, tend=None, check=False):
    if noverlap is None: #かぶる長さの設定
        noverlap = nfft//2 

    if tstart is not None: #スタート時間設定
        indx = np.where(t >= tstart)[0]
        istart = indx[0] 

    if tend is not None: #エンド時間設定
        if tend >= t[-1]:
            iend = len(t)
        else:
            indx = np.where(t > tend)[0]
            iend = indx[0]
   
    if iend == -1:
        iend = len(t)

    nsig = iend - istart   #計算する配列の長さの設定
    step = nfft - noverlap   #被らない配列数

    if check : #どの配列を何番目に計算するかを設定
        indices = np.arange(0, nsig-nfft+1, step) + istart
        for k, ind in enumerate(indices):
            print("{0}: {1}---{2}".format(k,t[ind],t[ind+nfft-1]))

    return (nsig-nfft)//step + 1


def running(x, t, dt=None, nfft=256, noverlap=None, istart=0, iend=-1, tstart=None, tend=None, window='hanning', detrend='constant'):
    """
    Estimate temporal evolution of power spectral density (running FFT) by dividing 
    the data into segments, computing a modified periodogram for each segment.
    ----------
    x : 1D-array
        Time series of measurement value
    t : 1D-array
        Time for x
    dt : float, optional
        Sampling time of the `x` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    iend: int, optional
        Last index in the specified axis of sigs. Defaults to -1.
    tend: float, optional
        End time of sigs. If None, t[iend] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    tave : ndarray
        Array of sample averaged times.
    Pxx : ndarray (2D)
        Power spectral density or power spectrum of x. Pxx(f, tave).
    See Also
    --------
    Notes
    -----
        see welch
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
    if dt is None: #時間配列の設定
        dt = (t[-1]-t[0]) / (len(t)-1) 

    if tstart is not None: #スタート時間の設定
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if tend is not None: #終了時間の設定
        if tend >= t[-1]:
            iend = len(t)
        else:
            indx = np.where(t > tend)[0]
            iend = indx[0] 
   
    if iend == -1:
        iend = len(t)
    #ある時間でのFFTを設定 時間分割したフーリエ成分
    f, tave, Pxx = dsp.spectrogram(x[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=True, scaling='density', axis=0, mode='psd')
    tave = tave + t[0] - dt
    return f, tave, Pxx

def csd(sigs, t, ref, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hanning', detrend='constant'):

    """
    Estimate cross power spectral density using Welch's method.
    Welch's method [1]_ computes an estimate of the power spectral density
    by dividing the data into overlapping segments, computing a modified
    periodogram for each segment and averaging the periodograms.
    Parameters
    ----------
    sigs : array_like
        Time series of measurement values
    t : 1D-array
        Time for sigs
    ref : 1D-array
        Time series of reference value
    dt : float, optional
        Sampling time of the `x` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap: int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nfft / 2``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window :  str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    Pxy : ndarray
        Cross power spectral density or cross power spectrum between ref and sigs.
    Pyy : ndarray
        Power spectral density or power spectrum of sigs.
    Pxx : ndarray
        Power spectral density or power spectrum of reference.
    See Also
    --------
    psd:power spectrum density
    running: running FFT
    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements.  For the default 'hanning' window an
    overlap of 50% is a reasonable trade off between accurately estimating
    the signal power, while not over counting any of the data.  Narrower
    windows may require a larger overlap.
    If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.
    .. versionadded:: 0.12.0
    References
    ----------
    .. [1] P. Welch, "The use of the fast Fourier transform for the
           estimation of power spectra: A method based on time averaging
           over short, modified periodograms", IEEE Trans. Audio
           Electroacoust. vol. 15, pp. 70-73, 1967.
    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika, vol. 37, pp. 1-16, 1950.
    """
   
    if dt is None: 
        dt = (t[-1]-t[0]) / (len(t)-1) 

    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
        
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart

    f, Pxy = dsp.csd(sigs[istart:iend,...], ref[istart:iend], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None,
                     detrend=detrend, return_onesided=False, scaling='density', axis=0)
    _, Pyy = dsp.welch(sigs[istart:iend, ...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None, detrend=detrend,
                       return_onesided=False, scaling='density', axis=0)
    _, Pxx = dsp.welch(ref[istart:iend], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=None, detrend=detrend,
                       return_onesided=False, scaling='density', axis=0)
    f = fft.fftshift(f)
    Pxx = fft.fftshift(Pxx)
    Pyy = fft.fftshift(Pyy,axes=0)
    Pxy = fft.fftshift(Pxy,axes=0)
    return f, Pxy, Pyy, Pxx

def xcoh2(Pxy, Pyy, Pxx):
    pwr = (Pxy*Pxy.conj()).real
    coh2 = pwr/Pyy
    
    if Pxy.ndim == 1:
        return coh2/Pxx
    
    for j in range(coh2.shape[1]):
       coh2[:,j] = coh2[:,j]/Pxx[:]
       
    return coh2

def xphase(Pxy, normalize=False, unsigned=False):
    phi = np.angle(Pxy)

    if unsigned:
       indx = np.where(phi < 0.0)
       phi[indx] = phi[indx] + 2*np.pi

    if normalize:
       phi = phi/(np.pi*2)
    
    return phi

def normalize(data, axis=0):
    data = np.asarray(data)
    ave = np.expand_dims(np.mean(data, axis), axis)
    ret = (data - ave)/ave        
    return ret

def psd2d(sigs, t, x, dt=None, dx=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hanning', detrend='constant'):
    """
    Estimate 2D cross power spectral density using Welch's method.
    ----------
    sigs : 2D-array
        Time series of measurement values
    t : 1D-array
        Time for sigs
    x : 1D-array
        Locations for sigs
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    dx : float, optional
        Sampling space of the `sigs` time series in units of m. Defaults
        to None.
    nfft : int, optional
        Length of each segment and  Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    Returns
    -------
    ft : ndarray
        Array of sample frequencies in time domain.
    fx : ndarray
        Array of sample frequencies in space domain.
    Ptx : ndarray (2D)
        Power spectral density or power spectrum of x. Pxx(f, tave).
    See Also
    --------
    Notes
    -----
        see welch
    """

    nt = len(t)
    nx = len(x)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if dx is None: 
        dx = (x[-1]-x[0]) / (nx-1) 

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    Ptx = np.zeros((nfft, nx))

    if sigs.size == 0:
        return np.empty(nfreq), tave, Pxx

    win_t = dsp.get_window(window, nfft)
    scale_t = dt / (win_t*win_t).sum()

    win_x = dsp.get_window(window, nx)    
    scale_x = dx / (win_x*win_x).sum()
    scale = scale_t*scale_x

    if not detrend:
        detrend_func = lambda seg: seg
    elif not hasattr(detrend, '__call__'):
        detrend_func = lambda seg: dsp.detrend(seg, type=detrend)
    else:
        detrend_func = detrend

    work = np.empty((nfft, nx), dtype=complex)
    spc = np.zeros((nfft, nx), dtype=complex)
    for ne in range(nensemble):
        i0 = istart + noverlap*ne
        i1 = i0 + nfft
        for j in range(nx):
            work[:,j] = win_x[j]*win_t*detrend_func(sigs[i0:i1,j])
        spc = fft.fft2(work)
        Ptx = Ptx + (spc * spc.conj()).real
    Ptx = scale*Ptx/nensemble
    Ptx = fft.fftshift(Ptx)
    ft = fft.fftfreq(nfft, dt)
    ft = fft.fftshift(ft)
    fx = fft.fftfreq(nx, dx)
    fx = fft.fftshift(fx)
                   
    return ft, fx, Ptx

def bicoherence_simple(s1, s2, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hanning', detrend='constant',vood=0.0):
    """
    Compute the bicoherence between two signals of the same lengths s1 and s2
    using the function scipy.signal.spectrogram.
    This calculates bi-coherence as defined and thus is not optimized.
    ----------
    s1, s2 : 1D-array
        Time series of measurement values should have identical dimensions
    t : 1D-array
        Time for s1 and s2
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    Returns
    -------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bi-coherence of sig. bicoh2(f1, f2).
    See Also
    --------
    Notes
    -----
        see scipy.signal.spectrogram
    """

    nt = len(t)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    # compute the stft
    f, t, spec1 = dsp.spectrogram(s1[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                  return_onesided=False, scaling='density', axis=0, mode='complex')
    _, _, spec2 = dsp.spectrogram(s2[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=False, scaling='density', axis=0, mode='complex')

    # transpose (f, t) -> (t, f)
    spec1 = np.transpose(spec1, [1, 0])
    spec2 = np.transpose(spec2, [1, 0])

    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    out_shape = list(spec1.shape)
    out_shape[0] = nhalf
    out_shape[1] = nf
    num = np.ones(out_shape)*vood
    denum = np.ones(out_shape)
        
    for i in range(nhalf):
        for j in range(nf):
            k = i + j
            if j < nhalf :
                if k <= nhalf-1:
                    num[i,j,...] =  np.abs(np.mean(spec1[:,i,...] * spec1[:,j,...] * np.conjugate(spec2[:,k,...]), axis=0))**2
                    denum[i,j,...] = np.mean(np.abs(spec1[:,i,...] * spec1[:,j,...])**2, axis=0) * np.mean(np.abs(np.conjugate(spec2[:,k,...]))**2, axis=0)
                continue
            if k >= nf :
                k = k - nf
            num[i,j,...] =  np.abs(np.mean(spec1[:,i,...] * spec1[:,j,...] * np.conjugate(spec2[:,k,...]), axis=0))**2
            denum[i,j,...] = np.mean(np.abs(spec1[:,i,...] * spec1[:,j,...])**2, axis=0) * np.mean(np.abs(np.conjugate(spec2[:,k,...]))**2, axis=0)
    bicoh2 = num / denum
                    
    bicoh2 = fft.fftshift(bicoh2, axes = 1)
    f1 = f[0:nhalf]
    f2 = fft.fftshift(f)
    return f1, f2, bicoh2

def abicoh2(sig, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None, window='hanning', detrend='constant',vood=0.0,axisf1=0):
    """
    Compute the squared-auto-bicoherence between two signals of the same lengths s1 and s2
    using the function scipy.signal.spectrogram
    ----------
    sig : 1D-array
        Time series of measurement values
    t : 1D-array
        Time for sig
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    axisf1 : int, optional
        Axis for f1. Defaults to 0. If axisf1 /= 0, axises [0, 1] of bicoh2 are transposed.
    Returns
    -------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bicoherence of sig. bicoh2(f1, f2).
    See Also
    --------
    Notes
    -----
        see scipy.signal.spectrogram
    """

    nt = len(t)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    # compute the stft
    f, t, spec = dsp.spectrogram(sig[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                  return_onesided=False, scaling='density', axis=0, mode='complex')

    # transpose (f, t) -> (t, f)
    spec = np.transpose(spec, [1, 0])

    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    fcol = np.arange(nf, dtype=int)
    lrow = np.arange(nhalf, dtype=int)
    lrow = np.roll(lrow,1)
    ha = la.hankel(fcol,lrow)
    num = np.abs(
                 np.mean(spec[:, fcol, None] * spec[:, None, lrow] * np.conjugate(spec[:, ha]), axis=0)
                ) ** 2
    denum = np.mean(
                    np.abs(spec[:, fcol, None] * spec[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec[:, ha])) ** 2, axis=0
                    )
    bicoh2 = num / denum

    for i in range(nhalf):
        for j in range(nhalf):
            k = i + j
            if k > nhalf-1:
                bicoh2[i,j,...] = vood    
               
    bicoh2 = fft.fftshift(bicoh2, axes = 0)
    if axisf1 == 0 :
        bicoh2 = np.transpose(bicoh2, [1, 0])
    f1 = f[0:nhalf]
    f2 =  fft.fftshift(f)
    return f1, f2, bicoh2

def xbicoh2(s1, s2, t, dt=None, nfft=256, noverlap=None, nensemble=1, istart=0, tstart=None,  comb='112', window='hanning', detrend='constant',vood=0.0,axisf1=0):
    """
    Compute the squared cross-bicoherence between two signals of the same lengths s1 and s2
    using the function scipy.signal.spectrogram
    ----------
    s1, s2 : 1D-array
        Time series of measurement values should have identical dimensions
    t : 1D-array
        Time for s1 and s2
    dt : float, optional
        Sampling time of the `sigs` time series in units of sec. Defaults
        to None.
    nfft : int, optional
        Length of the FFT used. 
    noverlap : int, optional
        Number of points to overlap between segments. If None,
        ``noverlap = nperseg // 8``.  Defaults to None.
    nensemble: int, optional
        Number of ensembles  Defaults to 1.
    istart: int, optional
        Start index in the specified axis of sigs. Defaults to 0.
    tstart: float, optional
        Start time of sigs. If None, t[istart] is used. Defaults to None.
    comb: str, optional
        Deefinition of cross-bicoherence.
        "112": B(f1,f2) = S1(f1)S1(f2)S2^(f1+f2)
        "121": B(f1,f2) = S1(f1)S2(f2)S1^(f1+f2) 
        "122": B(f1,f2) = S1(f1)S2(f2)S2^(f1+f2) 
    window : str or tuple or array_like, optional
        Desired window to use. See `get_window` for a list of windows and
        required parameters. Defaults to 'hanning'.
    detrend : str or function or False, optional
        Specifies how to detrend each segment. If `detrend` is a string,
        it is passed as the ``type`` argument to `detrend`.  If it is a
        function, it takes a segment and returns a detrended segment.
        If `detrend` is False, no detrending is done.  Defaults to 'constant'.
    vood : float, optional
        Value of out of domain (f1 + f2 > fnyq). Defaults to 0.
    axisf1 : int, optional
        Axis for f1. Defaults to 0. If axisf1 /= 0, axises [0, 1] of bicoh2 are transposed.
    Returns
    -------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bi-coherence of sig. bicoh2(f1, f2).
    See Also
    --------
    Notes
    -----
        see scipy.signal.spectrogram
    """

    nt = len(t)
    if dt is None: 
        dt = (t[-1]-t[0]) / (nt-1)

    if len(t) < nfft:
        warnings.warn('nfft = {0}, is greater than len(t) = {1}, using nfft = {2}'.format(nfft, nt, nt))
        nfft = len(t)
        
    if tstart is not None:
        indx = np.where(t >= tstart)[0]
        istart = indx[0]

    if noverlap is None:
        noverlap = nfft // 2
    elif noverlap >= nfft:
        raise ValueError('noverlap must be less than nfft.')
    
    step = nfft - noverlap
    iend = step*(nensemble-1) + nfft + istart
    if iend > nt:
        nnew = (nt-nfft-istart)/step + 1  
        warnings.warn('nensemble = {0} exceed boundary, using nensemble = {1}'.format(nt,nnew))
        nfft = nnew        

    # compute the stft
    f, t, spec1 = dsp.spectrogram(s1[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                  return_onesided=False, scaling='density', axis=0, mode='complex')
    _, _, spec2 = dsp.spectrogram(s2[istart:iend,...], fs=1.0/dt, window=window, nperseg=nfft, noverlap=noverlap, nfft=nfft, detrend=detrend,
                                   return_onesided=False, scaling='density', axis=0, mode='complex')

    # transpose (f, t) -> (t, f)
    spec1 = np.transpose(spec1, [1, 0])
    spec2 = np.transpose(spec2, [1, 0])

    # compute the bicoherence
    nf = f.size
    if nf % 2 == 0 :
       nhalf = nf//2
    else:
       nhalf = (nf+1)//2
    fcol = np.arange(nf, dtype=int)
    lrow = np.arange(nhalf, dtype=int)
    lrow = np.roll(lrow,1)
    ha = la.hankel(fcol,lrow)

    X = spec1
    if comb == "121":
        Y = spec2
        Z = spec1
    elif comb == "122":
        Y = spec2
        Z = spec2
    else:       
        Y = spec1
        Z = spec2
        
    num = np.abs(
                 np.mean(X[:, fcol, None] * Y[:, None, lrow] * np.conjugate(Z[:, ha]), axis=0)
#                 np.mean(spec1[:, fcol, None] * spec1[:, None, lrow] * np.conjugate(spec2[:, ha]), axis=0)
                ) ** 2
    denum = np.mean(
                    np.abs(X[:, fcol, None] * Y[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(Z[:, ha])) ** 2, axis=0
#                    np.abs(spec1[:, fcol, None] * spec1[:, None, lrow]) ** 2, axis=0) * np.mean(np.abs(np.conjugate(spec2[:, ha])) ** 2, axis=0
                    )
    bicoh2 = num / denum

    for i in range(nhalf):
        for j in range(nhalf):
            k = i + j
            if k > nhalf-1:
                bicoh2[i,j,...] = vood    
               
    bicoh2 = fft.fftshift(bicoh2, axes = 0)
    if axisf1 == 0 :
        bicoh2 = np.transpose(bicoh2, [1, 0])
    f1 = f[0:nhalf]
    f2 =  fft.fftshift(f)
    return f1, f2, bicoh2

def summedbicoh2(bicoh2, f1, f2, axisf1=0):
    """
    Compute the summed bicoherence
    ----------
    f1 : ndarray
        Array of sample frequencies from 0 to fnyq.
    f2 : ndarray
        Array of sample frequencies from -fnyq to fnyq.
    bicoh2 : ndarray (2D)
        Squared bi-coherence. bicoh2(f1, f2).
    Returns
    -------
    summed : ndarray
        Summed bi-coherence
    """

    nf = f1.size
    out_shape = list(bicoh2.shape)
    if axisf1 == 0 :
        del out_shape[0]
    else :
        del out_shape[1]
    out_shape[0] = nf
    summed = np.zeros(out_shape)
    nzero = f2.size - nf
       
    for k in range(nf):
        for i in range(nf):
            j = k - i + nzero
            if axisf1 == 0 :
                summed[k,...] = summed[k,...] + bicoh2[i,j,...]
            else:
                summed[k,...] = summed[k,...] + bicoh2[j,i,...]
                
    return summed

