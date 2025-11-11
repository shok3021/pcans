import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# 設定と定数
# =======================================================
# ★★★ 必ず各抽出スクリプトの出力ディレクトリに合わせてください ★★★
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fortran const モジュールからの値 (psd_extractor_revised.pyと一致させる)
GLOBAL_NX_PHYS = 320 # X方向セル数
GLOBAL_NY_PHYS = 639 # Y方向セル数
DELX = 1.0 # セル幅

X_HALF_LENGTH = GLOBAL_NX_PHYS * DELX / 2.0
X_MIN, X_MAX = -X_HALF_LENGTH, X_HALF_LENGTH
Y_MIN, Y_MAX = 0.0, GLOBAL_NY_PHYS * DELX


# =======================================================
# ヘルパー関数
# =======================================================

def load_2d_field_data(timestep, component):
    """
    電磁場データファイル (Bx, Exなど) を読み込む (extracted_data)
    """
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(FIELD_DATA_DIR, filename)

    try:
        data = np.loadtxt(filepath, delimiter=',')
        # ★★★ Ghostセル込みのサイズチェックを削除し、読み込んだデータをそのまま返す ★★★
        return data 
    except Exception:
        # ファイルが見つからない、またはサイズが合わない場合はゼロ配列を返す
        # (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS) のサイズを返すようにする
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_2d_moment_data(timestep, species, component):
    """
    モーメントデータファイル (density, Vxなど) を読み込む (extracted_psd_data_moments)
    """
    filename = f'data_{timestep}_{species}_{component}.txt'
    filepath = os.path.join(MOMENT_DATA_DIR, filename)
    
    try:
        data = np.loadtxt(filepath, delimiter=',')
        # 粒子モーメントは物理領域のみ (NY_PHYS, NX_PHYS) で保存されていることを想定
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
             # print(f"警告: {filepath} の形状 ({data.shape}) が期待値と異なります。")
             return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except Exception:
        # print(f"警告: {filepath} の読み込みに失敗しました。")
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))


def get_phys_data(raw_data):
    """電磁場データからGhostセルを除去する (電磁場ファイルがGhostセル込みで保存されている場合)"""
    # 物理領域は各次元のインデックス1から-2まで (2層のGhostセルを想定)
    # Fortranのインデックス nxgs=2, nxge=322 -> NumPyでは [1:-1]
    if raw_data.ndim == 2 and raw_data.shape[0] > GLOBAL_NY_PHYS and raw_data.shape[1] > GLOBAL_NX_PHYS:
        return raw_data[1:-1, 1:-1]
    return raw_data # 既に物理領域サイズの場合


# --- 計算関数 ---

def calculate_current_density(Bx, By, Ex, Ey, J_list, B0):
    """
    J_z = 密度 * (V_iz - V_ez) を計算 (プラズマ近似)
    J_x/J_y は計算に必要な情報がないため省略 (Maxwell方程式から計算するか、電場データを使用)
    ここでは簡易的に、Jz のみを計算し、Jx, Jyはゼロとする。
    
    Jz = (n_i * V_iz - n_e * V_ez) * q_e
    ただし、PSDデータから計算されたモーメントは通常 V_species (平均速度) です。
    電磁場シミュレーションでは、電流 J = n_e * q_e * (V_i - V_e) を使用します。
    ここでは、Jz は電子とイオンの平均速度 Vz を使用して計算します。
    
    PSDデータに含まれる密度は粒子数であり、規格化密度ではないため、ここでは簡略化。
    → Jz = (n_i * V_iz - n_e * V_ez) * 規格化定数
    → n_i = n_e = density_count (近似)
    """
    
    # 規格化された密度 n = density_count / (平均粒子数) は不明なため、
    # Jx = ne * (Vix - Vex), Jy = ne * (Viy - Vey), Jz = ne * (Viz - Vez) を近似します。
    # PSDから計算された density_count を電子とイオンで共通の密度プロキシとして使用します。
    
    n_e_count = J_list['density_count_e']
    n_i_count = J_list['density_count_i']
    Vx_e = J_list['Vx_e']
    Vx_i = J_list['Vx_i']
    Vy_e = J_list['Vy_e']
    Vy_i = J_list['Vy_i']
    Vz_e = J_list['Vz_e']
    Vz_i = J_list['Vz_i']

    # 局所密度を平均密度で規格化する近似
    N0 = (n_e_count + n_i_count) / 2.0
    avg_N0 = np.mean(N0[N0 > 1e-1]) # 空間的に非ゼロの領域の平均値
    if avg_N0 < 1: avg_N0 = 1 # ゼロ除算対策
    n_proxy = N0 / avg_N0
    
    # 規格化された電流 J / (n0 * q * VA) を計算
    J_x = n_proxy * (Vx_i - Vx_e)
    J_y = n_proxy * (Vy_i - Vy_e)
    J_z = n_proxy * (Vz_i - Vz_e)

    # 電子の圧力テンソル情報がないため、(d)パネルの計算は省略し、Ezをプロットします。
    Ez_non_ideal = Ez # 代用として Ez をそのまま使用

    return J_x, J_y, J_z, Ez_non_ideal


def calculate_magnetic_flux(Bx, By, DELX):
    """
    磁束関数 Psi を計算する (Bx = -d(Psi)/dy, By = d(Psi)/dx)
    Psi = - int(Bx dy) + int(By dx)
    ここでは By = d(Psi)/dx より、Psi_y = Psi_y + int(By dx) を使用
    """
    
    # cumtrapz: 台形則による累積積分
    
    # 1. By を x 方向に積分 (dPsi/dx) -> Psi(x, y) を求める
    # 積分方向: x (軸1)
    Psi_approx = cumtrapz(By, dx=DELX, axis=1, initial=0) # (NY, NX)
    
    # 積分定数 (yに依存する項) を決定するため、Psi(x=min) = 0 と仮定
    # Psi[:, 0] は既にゼロになっているはず
    
    return Psi_approx

def create_coordinates(NX, NY):
    """プロット用の X, Y 座標配列を作成する"""
    x = np.linspace(X_MIN, X_MAX, NX)
    y = np.linspace(Y_MIN, Y_MAX, NY)
    return np.meshgrid(x, y)

# =======================================================
# プロット関数
# =======================================================

def plot_single_panel(ax, X, Y, Z, Bx, By, title, label, cmap='RdBu_r', vmin=None, vmax=None):
    """単一パネルのプロット (Contourf + Streamlines)"""
    
    # 1. Contours (カラーマップ)
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
        
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, format='%.2f')
    cbar.set_label(label)
    
    # 2. Streamlines (磁力線: Bx, By)
    # データが密集しすぎているため、間引いてプロット
    stride_x = max(1, Bx.shape[1] // 30) # X方向に約30本
    stride_y = max(1, Bx.shape[0] // 30) # Y方向に約30本
    
    ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                  Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                  color='gray', linewidth=0.5, density=1.0, 
                  arrowstyle='-', minlength=0.1, zorder=1)
    
    # ラベルとタイトル
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)


def plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap='RdBu_r', vmin=None, vmax=None, stream_color='gray', stream_density=1.0):
    """統合パネル用のシンプルなプロット関数"""
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    
    # Streamlines
    stride_x = max(1, Bx.shape[1] // 15)
    stride_y = max(1, Bx.shape[0] // 15)
    ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                  Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                  color=stream_color, linewidth=0.5, density=stream_density, 
                  arrowstyle='-', minlength=0.1, zorder=1)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('$x/d_i$', fontsize=8)
    ax.set_ylabel('$y/d_i$', fontsize=8)
    ax.tick_params(direction='in', top=True, right=True, labelsize=7)
    return cf


# =======================================================
# メイン実行関数
# =======================================================

def main():
    if len(sys.argv) < 2:
        print("使用方法: python visualize_integrated.py [timestep] (例: 000500)")
        sys.exit(1)
        
    timestep = sys.argv[1]
    print(f"--- ターゲットタイムステップ: {timestep} ---")

    # --- 1. 必要なデータの読み込み ---
    print("データ読み込み中...")
    
    # (a) 電磁場データ (Ghostセル付きで抽出されている場合を想定し、読み込み後に物理領域を切り出す)
    Bx_raw = load_2d_field_data(timestep, 'Bx')
    By_raw = load_2d_field_data(timestep, 'By')
    Bz_raw = load_2d_field_data(timestep, 'Bz')
    Ex_raw = load_2d_field_data(timestep, 'Ex')
    Ey_raw = load_2d_field_data(timestep, 'Ey')
    Ez_raw = load_2d_field_data(timestep, 'Ez')
    
    # ★★★ デバッグのために、読み込んだ直後の形状を出力 ★★★
    print(f"DEBUG: Bx_raw shape: {Bx_raw.shape}")
    print(f"DEBUG: 期待される物理領域形状: ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS})")
    
    Bx = get_phys_data(Bx_raw)
    By = get_phys_data(By_raw)
    Bz = get_phys_data(Bz_raw)
    Ex = get_phys_data(Ex_raw)
    Ey = get_phys_data(Ey_raw)
    Ez = get_phys_data(Ez_raw)
    
    if Bx.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
        print("エラー: 読み込んだ電磁場データの形状が一致しません。抽出ステップを確認してください。")
        return
        
    # (b) 粒子モーメントデータ (物理領域のみで抽出されていることを想定)
    Vxe = load_2d_moment_data(timestep, 'electron', 'Vx')
    Vye = load_2d_moment_data(timestep, 'electron', 'Vy')
    Vze = load_2d_moment_data(timestep, 'electron', 'Vz')
    Vxi = load_2d_moment_data(timestep, 'ion', 'Vx')
    Vyi = load_2d_moment_data(timestep, 'ion', 'Vy')
    Vzi = load_2d_moment_data(timestep, 'ion', 'Vz')
    
    ne_count = load_2d_moment_data(timestep, 'electron', 'density_count')
    ni_count = load_2d_moment_data(timestep, 'ion', 'density_count')
    
    # --- 2. 派生量の計算 ---
    
    # (a) 磁束関数 Psi
    Psi = calculate_magnetic_flux(Bx, By, DELX)
    
    # (b) 電子の熱速度 (温度) (簡易計算)
    # 温度 (T) は圧力 (P) に比例し、圧力は PSD の速度分散に比例。
    # ここではデータがないため、簡易的に運動エネルギーの一部を代用する (精度は低い)
    # Te, Ti は Vx^2 + Vy^2 + Vz^2 の平均に比例すると仮定
    Te_proxy = Vxe**2 + Vye**2 + Vze**2
    Ti_proxy = Vxi**2 + Vyi**2 + Vzi**2
    
    # (c) 電流密度 J (Jx, Jy, Jz) と非理想Ez
    J_data = {'density_count_e': ne_count, 'density_count_i': ni_count,
              'Vx_e': Vxe, 'Vx_i': Vxi, 'Vy_e': Vye, 'Vy_i': Vyi,
              'Vz_e': Vze, 'Vz_i': Vzi}
              
    Jx, Jy, Jz, Ez_non_ideal_proxy = calculate_current_density(Bx, By, Ex, Ey, J_data, 1.0)
    
    # --- 3. 座標グリッドの作成 ---
    X, Y = create_coordinates(GLOBAL_NX_PHYS, GLOBAL_NY_PHYS)

    # --- 4. 可視化実行 ---

    # (A) Bxyz, Exyz, n_e, T_e, n_i, T_i, Psi, Jxyz, Ez_non_ideal のリストを準備
    plot_components = [
        # 1. Bxyz (磁力線は共通のBx, Byを使用)
        (Bx, 'Magnetic Field (Bx)', '$B_x/B_0$', plt.cm.RdBu_r),
        (By, 'Magnetic Field (By)', '$B_y/B_0$', plt.cm.RdBu_r),
        (Bz, 'Magnetic Field (Bz)', '$B_z/B_0$', plt.cm.RdBu_r),
        
        # 2. Exyz
        (Ex, 'Electric Field (Ex)', '$E_x/B_0$', plt.cm.coolwarm),
        (Ey, 'Electric Field (Ey)', '$E_y/B_0$', plt.cm.coolwarm),
        (Ez, 'Electric Field (Ez)', '$E_z/B_0$', plt.cm.coolwarm),
        
        # 3. 電子/イオンの密度・温度
        (ne_count, 'Electron Density', '$n_e$ (Counts)', plt.cm.viridis),
        (Te_proxy, 'Electron Temperature (Proxy)', '$T_e$ (Proxy)', plt.cm.plasma),
        (ni_count, 'Ion Density', '$n_i$ (Counts)', plt.cm.viridis),
        (Ti_proxy, 'Ion Temperature (Proxy)', '$T_i$ (Proxy)', plt.cm.plasma),
        
        # 4. 磁束 Psi
        (Psi, 'Magnetic Flux $\Psi$', '$\Psi$', plt.cm.seismic),
        
        # 5. 電流密度 J (Jxyz)
        (Jx, 'Current Density (Jx)', '$J_x$', plt.cm.RdBu_r),
        (Jy, 'Current Density (Jy)', '$J_y$', plt.cm.RdBu_r),
        (Jz, 'Current Density (Jz)', '$J_z$', plt.cm.RdBu_r),
        
        # 6. 非理想 Ez (J_z, Psi と同じパネルに含めても良いが、ここでは独立)
        (Ez_non_ideal_proxy, 'Non-Ideal Electric Field ($E_{||}$ Proxy)', '$(E_z + V_e \times B_z)$ Proxy', plt.cm.jet),

    ]

    # --- プロット A: 各成分を個別の図に出力 ---
    print("個別のプロットを生成中...")
    for i, (Z, title, label, cmap) in enumerate(plot_components):
        if i >= 13: # Jxyzの3成分以降は最後の統合パネルに含める
             break

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_single_panel(ax, X, Y, Z, Bx, By, f"Timestep {timestep}: {title}", label, cmap=cmap, 
                          vmin=-np.nanmax(np.abs(Z)), vmax=np.nanmax(np.abs(Z)))
        
        fig.tight_layout()
        output_filename = os.path.join(OUTPUT_DIR, f'plot_{timestep}_{label.replace("/", "_").replace("$", "").replace(" ", "_")}.png')
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)
        print(f"-> {title} を {output_filename} に保存しました。")
    
    # --- プロット B: 全ての重要な要素を1枚の図に出力 ---
    print("\n統合パネルを生成中...")
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), sharex=True, sharey=True)
    ax_list = axes.flatten()
    
    # 統合パネル用のプロットリスト (全16枚)
    combined_plots = [
        # 1. 磁場 B
        (Bx, 'a) $B_x$', '$B_x/B_0$', plt.cm.RdBu_r),
        (By, 'b) $B_y$', '$B_y/B_0$', plt.cm.RdBu_r),
        (Bz, 'c) $B_z$', '$B_z/B_0$', plt.cm.RdBu_r),
        
        # 2. 電場 E
        (Ex, 'd) $E_x$', '$E_x/B_0$', plt.cm.coolwarm),
        (Ey, 'e) $E_y$', '$E_y/B_0$', plt.cm.coolwarm),
        (Ez, 'f) $E_z$', '$E_z/B_0$', plt.cm.coolwarm),
        
        # 3. 密度 n (電子)
        (ne_count, 'g) $n_e$ (Density)', '$n_e$ Count', plt.cm.viridis),
        
        # 4. 温度 T (電子)
        (Te_proxy, 'h) $T_e$ (Proxy)', '$T_e$ Proxy', plt.cm.plasma),
        
        # 5. 磁束 Psi
        (Psi, 'i) $\Psi$', '$\Psi$', plt.cm.seismic),

        # 6. 電流密度 J
        (Jx, 'j) $J_x$', '$J_x$', plt.cm.RdBu_r),
        (Jy, 'k) $J_y$', '$J_y$', plt.cm.RdBu_r),
        (Jz, 'l) $J_z$', '$J_z$', plt.cm.RdBu_r),

        # 7. その他の重要量 (イオン密度、イオン温度、非理想 Ez)
        (ni_count, 'm) $n_i$ (Density)', '$n_i$ Count', plt.cm.viridis),
        (Ti_proxy, 'n) $T_i$ (Proxy)', '$T_i$ Proxy', plt.cm.plasma),
        (Ez_non_ideal_proxy, 'o) Non-Ideal $E_z$ Proxy', '$E_{||}$ Proxy', plt.cm.jet),
        
        # 最後のパネルは空欄または何か追加のプロット
        (Bx, 'p) Bx (Repeat)', '$B_x/B_0$', plt.cm.RdBu_r),

    ]
    
    for i, (Z, title, label, cmap) in enumerate(combined_plots):
        ax = ax_list[i]
        # vmin/vmaxを対称にするために最大絶対値を使用
        max_abs = np.nanmax(np.abs(Z)) 
        
        cf = plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap=cmap, vmin=-max_abs, vmax=max_abs)
        
        # カラーバーのサイズ調整 (全てのサブプロットにカラーバーを入れると窮屈になるため、省略)

    fig.tight_layout()
    output_filename_combined = os.path.join(OUTPUT_DIR, f'plot_combined_{timestep}.png')
    plt.savefig(output_filename_combined, dpi=300)
    print(f"-> 全てを含む統合パネルを {output_filename_combined} に保存しました。")
    plt.show()

# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    # Matplotlibのフォント設定
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    main()