import numpy as np
from scipy.fft import fft, ifft, fftshift, fftfreq

def fftf(data, time, direction=-1):
    """
    IDLのfftf.proのPython (Scipy) 版
    direction: -1 (順変換), +1 (逆変換)
    """
    
    n = len(data)
    if n != len(time):
        raise ValueError("data and time must have the same length")
        
    dt = time[1] - time[0]
    
    # 周波数軸 (シフト前)
    freq = fftfreq(n, dt)
    
    # IDLの fft(..., -1) と fft(..., 1) は規格化なし
    # norm="backward" (規格化なし) を指定
    if direction == -1:
        ans = fft(data, norm="backward")
        # IDLの規格化
        T_idl = np.max(time) - np.min(time)
        ans = ans * T_idl
    elif direction == +1:
        ans = fft(data, direction=1, norm="backward") # 逆変換 (規格化なし)
        # IDLの規格化
        T_idl = np.max(time) - np.min(time)
        ans = ans * T_idl / n
    else:
        raise ValueError("direction must be -1 or +1")

    # DC成分を中央にシフト
    ans_shifted = fftshift(ans)
    freq_shifted = fftshift(freq)
    
    return ans_shifted, freq_shifted