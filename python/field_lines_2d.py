import numpy as np
from scipy.interpolate import RegularGridInterpolator

def field_lines_2d(ux, uy, nsteps=10000, length=1.0, r0=None, npos=100, direction=1.0):
    """
    IDLの field_lines_2d.pro のPython (Numpy/Scipy) 版。
    ベクトル場 (ux, uy) の流線をトレースします。

    Args:
        ux (np.ndarray): ベクトル場のX成分。形状は (nx, ny) を想定。
        uy (np.ndarray): ベクトル場のY成分。形状は (nx, ny) を想定。
        nsteps (int): トレースの最大ステップ数。
        length (float): 各ステップの固定長 (IDLの 'len')。
        r0 (np.ndarray): 初期位置 (2, npos)。指定しない場合はランダム。
        npos (int): r0 がない場合に生成する流線の数。
        direction (float): トレース方向 (1.0: 順方向, -1.0: 逆方向)。

    Returns:
        np.ndarray: トレース結果の配列 (2, nsteps, npos)。
    """
    
    # --- 1. 入力データの形状と座標軸のセットアップ ---
    # IDLの (nx, ny) 規約に従う
    if ux.shape != uy.shape:
        raise ValueError("ux と uy は同じ形状である必要があります。")
    
    nx, ny = ux.shape
    x_coords = np.arange(nx)
    y_coords = np.arange(ny)

    # --- 2. 補間器のセットアップ ---
    # IDL: cubic=-0.5 -> method='cubic'
    # IDL: デフォルト (境界の外挿) -> bounds_error=False, fill_value=None
    try:
        interp_ux = RegularGridInterpolator(
            (x_coords, y_coords), ux, 
            method='cubic', bounds_error=False, fill_value=None
        )
        interp_uy = RegularGridInterpolator(
            (x_coords, y_coords), uy, 
            method='cubic', bounds_error=False, fill_value=None
        )
    except ValueError as e:
        print(f"補間器の初期化に失敗しました: {e}")
        print("入力 'ux', 'uy' の形状が (nx, ny) であることを確認してください。")
        return None

    # --- 3. 初期位置 (r0) のセットアップ ---
    if r0 is None:
        rng = np.random.default_rng()
        r0 = np.zeros((2, npos))
        # IDL: randomu(seed, npos) * (nx-1)
        r0[0, :] = rng.random(npos) * (nx - 1)
        r0[1, :] = rng.random(npos) * (ny - 1)
    else:
        if r0.ndim != 2 or r0.shape[0] != 2:
            raise ValueError(f"r0 の形状は (2, npos) である必要がありますが、{r0.shape} です。")
        npos = r0.shape[1]
        
    # --- 4. 出力配列とアクティブマスクの初期化 ---
    # IDL: r = dblarr(2, nsteps, npos)
    r = np.zeros((2, nsteps, npos), dtype=np.float64)
    r[:, 0, :] = r0

    # 現在計算中の（境界内の）流線を追跡するマスク
    active = np.ones(npos, dtype=bool)

    # --- 5. 流線のトレース (ステップ 'i' でループ) ---
    for i in range(1, nsteps):
        # アクティブな流線がなくなったら終了
        if not np.any(active):
            # IDLのロジック: 停止したステップ 'i-1' の位置で残りを埋める
            # (r[:, i:, :] = r[:, i-1:i, :, :] と等価)
            r[:, i:, :] = r[:, i-1, np.newaxis, :]
            break
        
        # 現在アクティブな流線の「前ステップ(i-1)」の位置を取得
        r_prev_active = r[:, i-1, active] # (2, n_active)
        
        # (n_active, 2) の形状で補間器に渡す
        points_to_interp = r_prev_active.T
        
        # ベクトル場で補間
        utx = interp_ux(points_to_interp)
        uty = interp_uy(points_to_interp)

        # 大きさを計算 (ゼロ除算防止)
        uabs = np.sqrt(utx**2 + uty**2)
        uabs[uabs == 0.0] = 1e-10
        
        # 正規化
        utx_norm = utx / uabs
        uty_norm = uty / uabs
        
        # オイラーステップで新しい位置 r(i) を計算
        r_new_x = r_prev_active[0, :] + length * direction * utx_norm
        r_new_y = r_prev_active[1, :] + length * direction * uty_norm
        
        # --- 6. 境界チェック (IDL: 0 < r < N-1) ---
        flags_x = (r_new_x > 0) & (r_new_x < (nx - 1))
        flags_y = (r_new_y > 0) & (r_new_y < (ny - 1))
        
        in_bounds = flags_x & flags_y # (n_active,) のブール配列
        
        # 新しい位置を 'r' 配列の [i, active] の位置に格納
        # (この時点では境界外の位置も含む)
        r[0, i, active] = r_new_x
        r[1, i, active] = r_new_y
        
        # --- 7. 停止した流線の処理 ---
        
        # 停止した流線 (active だったが in_bounds でなくなった)
        stopped_this_step_mask = ~in_bounds # (n_active,)
        
        # 'npos' 全体でのインデックスを取得
        active_indices = np.where(active)[0]
        idx_stopped = active_indices[stopped_this_step_mask]
        
        if len(idx_stopped) > 0:
            # 停止した流線の「境界外に出た位置 r(i)」を取得
            r_stop = r[:, i, idx_stopped] # (2, n_stopped)
            
            # IDLのロジック: r(istop+1:*, k) = r(istop, k)
            # r(i+1) 以降を、r(i) の位置で埋める
            # (2, 1, n_stopped) を (2, nsteps-i-1, n_stopped) にブロードキャスト
            r[:, i+1:, idx_stopped] = r_stop[:, np.newaxis, :]

            # 'active' マスクを更新
            active[idx_stopped] = False

    return r