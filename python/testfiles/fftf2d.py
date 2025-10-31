import numpy as np
from scipy.fft import fft2, fftshift, fftfreq

def fftf2d(data, x, t, direction=-1):
    """
    IDLのfft2d.proのPython (Scipy) 版
    data: (nx, nt) 配列
    x: 1D配列 (長さ nx)
    t: 1D配列 (長さ nt)
    """
    
    nx, nt = data.shape
    if len(x) != nx or len(t) != nt:
        raise ValueError("Data dimensions do not match x and t lengths")

    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # 波数軸 (x方向) と 周波数軸 (t方向)
    wnum = fftfreq(nx, dx)
    freq = fftfreq(nt, dt)
    
    # 規格化定数
    Tx_idl = np.max(x) - np.min(x)
    Tt_idl = np.max(t) - np.min(t)
    
    # IDLの fft(..., -1) と fft(..., 1) は規格化なし
    if direction == -1:
        ans = fft2(data, norm="backward")
        ans = ans * Tx_idl * Tt_idl
    elif direction == +1:
        ans = fft2(data, direction=1, norm="backward") # 逆変換 (規格化なし)
        ans = ans * Tx_idl * Tt_idl / (nx * nt)
    else:
        raise ValueError("direction must be -1 or +1")

    # DC成分を中央にシフト
    # IDLの reverse(shift(...)) は、Pythonの fftshift(ans) と等価
    ans_shifted = fftshift(ans)
    
    # 軸もシフト
    wnum_shifted = fftshift(wnum)
    freq_shifted = fftshift(freq)
    
    return ans_shifted, wnum_shifted, freq_shifted