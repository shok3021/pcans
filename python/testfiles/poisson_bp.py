import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import solve_banded

def gssel_scipy(dm, p_j, nx):
    """
    IDLのgsselをscipy.linalg.solve_bandedで実装。
    IDLコードの境界条件 (-2*dm-1.0) を再現。
    """
    # (1, 1) は、主対角の上(1)と下(1)に非ゼロ対角があることを示す
    ab = np.zeros((3, nx), dtype=complex)
    
    # 上対角 (ab[0, 1:])
    ab[0, 1:] = 1.0
    # 主対角 (ab[1, :])
    ab[1, :] = -2.0 * dm
    ab[1, 0] -= 1.0
    ab[1, -1] -= 1.0
    # 下対角 (ab[2, :-1])
    ab[2, :-1] = 1.0
    
    return solve_banded((1, 1), ab, p_j)

def poisson_bp(b):
    """
    IDLのpoisson_bpのPython (Scipy) 版
    b: (nx, ny) の入力配列
    """
    nx, ny = b.shape
    x = np.zeros((nx, ny), dtype=complex)
    
    # Y方向 (axis=1) に FFT
    # IDLの fft(..., -1) は順変換 (規格化なし)
    # scipy.fft.fft(..., norm="backward") も規格化なし
    p = fft(b, axis=1, norm="backward")
    
    # 各波数 j (ky) ごとに 1D ヘルムホルツ方程式を解く
    
    # j=0
    j = 0
    dm = 1.0
    x[:, j] = gssel_scipy(dm, p[:, j], nx)
    
    # j=ny/2 (ナイキスト周波数)
    j = ny // 2
    dm = 1.0 + 2.0 * np.sin(np.pi / 2.0)**2 
    x[:, j] = gssel_scipy(dm, p[:, j], nx)
    
    # j=1 から ny/2-1
    for j in range(1, ny // 2):
        argj = float(j) * np.pi / float(ny)
        dm = 1.0 + 2.0 * np.sin(argj)**2
        
        # j (正の周波数)
        x[:, j] = gssel_scipy(dm, p[:, j], nx)
        
        # jj (負の周波数)
        jj = ny - j
        x[:, jj] = gssel_scipy(dm, p[:, jj], nx)

    # Y方向 (axis=1) に 逆FFT
    # IDLの fft(..., 1) も規格化なし。
    # scipy.fft.ifft は 1/N 規格化を行う。
    # IDLとペアにするため、ifft(..., norm="backward") を使う
    phi = fft(x, axis=1, direction=-1, norm="backward") # これも規格化なし
    
    # IDLの fft(..., 1) が 1/N 規格化 *する* 場合 (HELPには書いていないが...)
    # phi = ifft(x, axis=1) # (1/N 規格化)
    
    # fft(..., -1) と fft(..., 1) がペアで使われているため、
    # Pythonでも fft(..., norm="backward") と ifft(..., norm="backward") (1/Nあり) のペアが正しい
    phi = ifft(x, axis=1, norm="backward") # 1/N 規格化あり (scipyのデフォルト)
    
    # b が実数なら、phi も実数のはず
    if np.isrealobj(b):
        return phi.real
    else:
        return phi