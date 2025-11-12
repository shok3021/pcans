import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# 設定と定数 (変更なし)
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
# ヘルパー関数 (load/get_phys_data は変更なし)
# =======================================================
# ... (load_2d_field_data, load_2d_moment_data, get_phys_data, calculate_current_density, calculate_magnetic_flux, create_coordinates は変更なし) ...

# 以前のコードの関数定義を再掲載
def load_2d_field_data(timestep, component):
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(FIELD_DATA_DIR, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data 
    except Exception:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_2d_moment_data(timestep, species, component):
    filename = f'data_{timestep}_{species}_{component}.txt'
    filepath = os.path.join(MOMENT_DATA_DIR, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
             return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except Exception:
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def get_phys_data(raw_data):
    return raw_data

def calculate_current_density(Bx, By, Ex, Ey, Ez, J_list, B0):
    n_e_count = J_list['density_count_e']
    n_i_count = J_list['density_count_i']
    Vx_e = J_list['Vx_e']
    Vx_i = J_list['Vx_i']
    Vy_e = J_list['Vy_e']
    Vy_i = J_list['Vy_i']
    Vz_e = J_list['Vz_e']
    Vz_i = J_list['Vz_i']

    N0 = (n_e_count + n_i_count) / 2.0
    avg_N0 = np.mean(N0[N0 > 1e-1]) 
    if avg_N0 < 1: avg_N0 = 1 
    n_proxy = N0 / avg_N0
    
    J_x = n_proxy * (Vx_i - Vx_e)
    J_y = n_proxy * (Vy_i - Vy_e)
    J_z = n_proxy * (Vz_i - Vz_e)
    Ez_non_ideal = Ez 
    return J_x, J_y, J_z, Ez_non_ideal

def calculate_magnetic_flux(Bx, By, DELX):
    Psi_approx = cumtrapz(By, dx=DELX, axis=1, initial=0)
    return Psi_approx

def create_coordinates(NX, NY):
    x = np.linspace(X_MIN, X_MAX, NX)
    y = np.linspace(Y_MIN, Y_MAX, NY)
    return np.meshgrid(x, y)

# --- ゼロデータに対するプロット範囲調整 (修正済みロジック) ---
def get_plot_range(Z):
    """ゼロ配列の場合にプロット範囲を調整"""
    max_abs = np.nanmax(np.abs(Z))
    if np.isclose(max_abs, 0.0):
        # データが完全にゼロの場合、微小な範囲を設定してプロットを強制
        return -1e-6, 1e-6
    return -max_abs, max_abs

# =======================================================
# プロット関数 (変更なし)
# =======================================================
# ... (plot_single_panel, plot_combined は変更なし) ...

def plot_single_panel(ax, X, Y, Z, Bx, By, title, label, cmap='RdBu_r', vmin=None, vmax=None):
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    cbar = plt.colorbar(cf, ax=ax, format='%.2f')
    cbar.set_label(label)
    stride_x = max(1, Bx.shape[1] // 30) 
    stride_y = max(1, Bx.shape[0] // 30) 
    ax.streamplot(X[::stride_y, ::stride_x], Y[::stride_y, ::stride_x], 
                  Bx[::stride_y, ::stride_x], By[::stride_y, ::stride_x], 
                  color='gray', linewidth=0.5, density=1.0, 
                  arrowstyle='-', minlength=0.1, zorder=1)
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(title)
    ax.tick_params(direction='in', top=True, right=True)

def plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap='RdBu_r', vmin=None, vmax=None, stream_color='gray', stream_density=1.0):
    if vmin is None: vmin = Z.min()
    if vmax is None: vmax = Z.max()
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
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
    # ... (中略)
    Bx_raw = load_2d_field_data(timestep, 'Bx')
    By_raw = load_2d_field_data(timestep, 'By')
    Bz_raw = load_2d_field_data(timestep, 'Bz')
    Ex_raw = load_2d_field_data(timestep, 'Ex')
    Ey_raw = load_2d_field_data(timestep, 'Ey')
    Ez_raw = load_2d_field_data(timestep, 'Ez')
    
    print(f"DEBUG: Bx_raw shape: {Bx_raw.shape}")
    print(f"DEBUG: 期待される物理領域形状: ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS})")
    
    Bx = get_phys_data(Bx_raw)
    By = get_phys_data(By_raw)
    Bz = get_phys_data(Bz_raw)
    Ex = get_phys_data(Ex_raw)
    Ey = get_phys_data(Ey_raw)
    Ez = get_phys_data(Ez_raw)
    
    if Bx.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
        print(f"エラー: 読み込んだ電磁場データの形状 ({Bx.shape}) が一致しません。期待される形状: ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS})。抽出ステップを確認してください。")
        return
        
    # (b) 粒子モーメントデータ (読み込み)
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
    Te_proxy = Vxe**2 + Vye**2 + Vze**2
    Ti_proxy = Vxi**2 + Vyi**2 + Vzi**2
    
    # (c) 電流密度 J (Jx, Jy, Jz) と非理想Ez
    J_data = {'density_count_e': ne_count, 'density_count_i': ni_count,
              'Vx_e': Vxe, 'Vx_i': Vxi, 'Vy_e': Vye, 'Vy_i': Vyi,
              'Vz_e': Vze, 'Vz_i': Vzi}
              
    Jx, Jy, Jz, Ez_non_ideal_proxy = calculate_current_density(Bx, By, Ex, Ey, Ez, J_data, 1.0)
    
    # --- 3. 座標グリッドの作成 ---
    X, Y = create_coordinates(GLOBAL_NX_PHYS, GLOBAL_NY_PHYS)

    # --- 4. 可視化実行 ---
    
    # プロットコンポーネントのリスト
    plot_components = [
        ('Bx', Bx, 'Magnetic Field (Bx)', '$B_x/B_0$', plt.cm.RdBu_r),
        ('By', By, 'Magnetic Field (By)', '$B_y/B_0$', plt.cm.RdBu_r),
        ('Bz', Bz, 'Magnetic Field (Bz)', '$B_z/B_0$', plt.cm.RdBu_r),
        
        ('Ex', Ex, 'Electric Field (Ex)', '$E_x/B_0$', plt.cm.coolwarm),
        ('Ey', Ey, 'Electric Field (Ey)', '$E_y/B_0$', plt.cm.coolwarm),
        ('Ez', Ez, 'Electric Field (Ez)', '$E_z/B_0$', plt.cm.coolwarm),
        
        ('ne', ne_count, 'Electron Density', '$n_e$ (Counts)', plt.cm.viridis),
        ('Te', Te_proxy, 'Electron Temperature (Proxy)', '$T_e$ (Proxy)', plt.cm.plasma),
        ('ni', ni_count, 'Ion Density', '$n_i$ (Counts)', plt.cm.viridis),
        ('Ti', Ti_proxy, 'Ion Temperature (Proxy)', '$T_i$ (Proxy)', plt.cm.plasma),
        
        ('Psi', Psi, 'Magnetic Flux $\Psi$', '$\Psi$', plt.cm.seismic),
        
        ('Jx', Jx, 'Current Density (Jx)', '$J_x$', plt.cm.RdBu_r),
        ('Jy', Jy, 'Current Density (Jy)', '$J_y$', plt.cm.RdBu_r),
        ('Jz', Jz, 'Current Density (Jz)', '$J_z$', plt.cm.RdBu_r),
        
        ('Ez_non_ideal', Ez_non_ideal_proxy, 'Non-Ideal Electric Field ($E_{||}$ Proxy)', '$(E_z + V_e \\times B_z)$ Proxy', plt.cm.jet),
    ]

    # --- プロット A: 各成分を個別のサブディレクトリに出力 ---
    print("個別のプロットを生成中 (サブディレクトリに保存)...")
    
    for tag, Z, title, label, cmap in plot_components:
        # サブディレクトリ名を決定 (例: Bx, Psi, ne など)
        SUB_DIR = os.path.join(OUTPUT_DIR, tag.replace('/', '_'))
        os.makedirs(SUB_DIR, exist_ok=True)

        # ゼロデータに対する範囲調整
        vmin, vmax = get_plot_range(Z)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_single_panel(ax, X, Y, Z, Bx, By, f"Timestep {timestep}: {title}", label, cmap=cmap, 
                          vmin=vmin, vmax=vmax)
        
        fig.tight_layout()
        # ファイル名をコンポーネントタグベースで作成
        output_filename = os.path.join(SUB_DIR, f'plot_{timestep}_{tag}.png')
        plt.savefig(output_filename, dpi=200)
        plt.close(fig)
        print(f"-> {title} を {output_filename} に保存しました。")
    
    # --- プロット B: 全ての重要な要素を1枚の統合パネルに出力 ---
    print("\n統合パネルを生成中 (ルートディレクトリに保存)...")
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
        (Ez_non_ideal_proxy, 'o) Non-Ideal $E_z$ Proxy', '$(E_z + V_e \\times B_z)$ Proxy', plt.cm.jet),
        
        # 最後のパネルは空欄または何か追加のプロット
        (Bx, 'p) Bx (Repeat)', '$B_x/B_0$', plt.cm.RdBu_r),

    ]
    
    for i, (Z, title, label, cmap) in enumerate(combined_plots):
        ax = ax_list[i]
        vmin, vmax = get_plot_range(Z)
        
        cf = plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.tight_layout()
    output_filename_combined = os.path.join(OUTPUT_DIR, f'plot_combined_{timestep}.png')
    plt.savefig(output_filename_combined, dpi=300)
    print(f"-> 全てを含む統合パネルを {output_filename_combined} に保存しました。")
    # plt.show() # 統合パネルのみ表示

# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    # Matplotlibのフォント設定
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    main()