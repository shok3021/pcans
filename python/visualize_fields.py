import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.integrate import cumtrapz

# =======================================================
# ヘルパー関数 (load_simulation_parameters, load_2d_field_data, etc.)
# =======================================================
# (前回のコードから変更なし)

def load_simulation_parameters(param_filepath):
    """
    init_param.dat (または同等のログファイル) を読み込み、
    光速 (c) と イオン・プラズマ周波数 (Fpi) を抽出する。
    """
    C_LIGHT = None
    FPI = None
    
    print(f"パラメータファイルを読み込み中: {param_filepath}")

    try:
        with open(param_filepath, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                
                # 'c' の値を抽出
                if stripped_line.startswith('dx, dt, c'):
                    try:
                        parts = stripped_line.split()
                        C_LIGHT = float(parts[6]) # 7番目の要素 (0-indexed)
                        print(f"  -> 'c' の値を検出: {C_LIGHT}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'c' の値の解析に失敗。行: {line}")
                        
                # 'Fpi' の値を抽出
                elif stripped_line.startswith('Fpe, Fge, Fpi Fgi'):
                    try:
                        parts = stripped_line.split()
                        FPI = float(parts[7]) # 8番目の要素 (0-indexed)
                        print(f"  -> 'Fpi' の値を検出: {FPI}")
                    except (IndexError, ValueError):
                        print(f"  -> エラー: 'Fpi' の値の解析に失敗。行: {line}")

    except FileNotFoundError:
        print(f"★★ エラー: パラメータファイルが見つかりません: {param_filepath}")
        print("     DIの計算に失敗しました。スクリプトを終了します。")
        sys.exit(1)
        
    if C_LIGHT is None or FPI is None:
        print("★★ エラー: ファイルから 'c' または 'Fpi' の値を抽出できませんでした。")
        print("     ファイルの内容を確認してください。スクリプトを終了します。")
        sys.exit(1)
        
    return C_LIGHT, FPI

# =======================================================
# 設定と定数
# =======================================================
# ( __file__ が未定義の場合のエラーを防ぐためのフォールバック)
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.') # (Jupyterなどでの実行用)

FIELD_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 
MOMENT_DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- init_param.dat のパスを指定 ---
# ★★★ 必要に応じてこのパスを修正してください ★★★
PARAM_FILE_PATH = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans/em2d_mpi/md_mrx/dat/init_param.dat') 

# --- パラメータの読み込みと di の計算 ---
# (ループ処理の前に一度だけ実行)
C_LIGHT, FPI = load_simulation_parameters(PARAM_FILE_PATH)
DI = C_LIGHT / FPI # イオンスキンデプス (di = c / omega_pi)

print(f"--- 規格化スケール: イオンスキンデプス d_i = {DI:.4f} (c={C_LIGHT}, Fpi={FPI}) ---")

# --- Fortran const モジュールからの値 ---
GLOBAL_NX_PHYS = 320 # X方向セル数
GLOBAL_NY_PHYS = 639 # Y方向セル数
DELX = 1.0 # セル幅

# =======================================================
# ヘルパー関数 (データ読み込み・計算)
# =======================================================
def load_2d_field_data(timestep, component):
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(FIELD_DATA_DIR, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS}) と異なります。ゼロ配列を返します。")
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data 
    except Exception as e:
        # ★ ループ処理のため、エラーがあっても停止させず、警告を出す
        print(f"警告: {filepath} の読み込みに失敗しました ({e})。ゼロ配列を返します。")
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

def load_2d_moment_data(timestep, species, component):
    filename = f'data_{timestep}_{species}_{component}.txt'
    filepath = os.path.join(MOMENT_DATA_DIR, filename)
    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.shape != (GLOBAL_NY_PHYS, GLOBAL_NX_PHYS):
            print(f"警告: {filepath} の形状 ({data.shape}) が期待値 ({GLOBAL_NY_PHYS}, {GLOBAL_NX_PHYS}) と異なります。ゼロ配列を返します。")
            return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))
        return data
    except Exception as e:
        # ★ ループ処理のため、エラーがあっても停止させず、警告を出す
        print(f"警告: {filepath} の読み込みに失敗しました ({e})。ゼロ配列を返します。")
        return np.zeros((GLOBAL_NY_PHYS, GLOBAL_NX_PHYS))

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
    # ゼロ割を避ける
    N0_filtered = N0[N0 > 1e-1]
    if len(N0_filtered) == 0:
        avg_N0 = 1.0 # データが全くない場合
    else:
        avg_N0 = np.mean(N0_filtered)
        
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
    x_phys = np.linspace(-GLOBAL_NX_PHYS * DELX / 2.0, GLOBAL_NX_PHYS * DELX / 2.0, NX)
    y_phys = np.linspace(0.0, GLOBAL_NY_PHYS * DELX, NY)
    x_norm = x_phys / DI
    y_norm = y_phys / DI
    return np.meshgrid(x_norm, y_norm)

def get_plot_range(Z):
    """ゼロ配列の場合にプロット範囲を調整"""
    try:
        max_abs = np.nanmax(np.abs(Z))
    except ValueError: # すべてNaNの場合
        max_abs = 0.0
        
    if np.isclose(max_abs, 0.0) or not np.isfinite(max_abs):
        return -1e-6, 1e-6
    return -max_abs, max_abs

# =======================================================
# プロット関数 (変更なし)
# =======================================================

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
# ★★★ メイン実行関数 (単一ステップ処理用) ★★★
# =======================================================

def process_timestep(timestep):
    """指定された単一のタイムステップのデータを処理・可視化する"""
    
    print(f"\n=============================================")
    print(f"--- ターゲットタイムステップ: {timestep} の処理開始 ---")
    print(f"=============================================")

    # --- 1. 必要なデータの読み込み ---
    print("電磁場データを読み込み中...")
    Bx = load_2d_field_data(timestep, 'Bx')
    By = load_2d_field_data(timestep, 'By')
    Bz = load_2d_field_data(timestep, 'Bz')
    Ex = load_2d_field_data(timestep, 'Ex')
    Ey = load_2d_field_data(timestep, 'Ey')
    Ez = load_2d_field_data(timestep, 'Ez')

    # Bxがゼロ配列（読み込み失敗）の場合、形状チェックが失敗する可能性がある
    # 読み込み失敗の時点でゼロ配列が返るので、形状チェックは不要
    # (load_2d_field_data内でチェック＆警告するように変更済み)
        
    print("粒子モーメントデータを読み込み中...")
    Vxe = load_2d_moment_data(timestep, 'electron', 'Vx')
    Vye = load_2d_moment_data(timestep, 'electron', 'Vy')
    Vze = load_2d_moment_data(timestep, 'electron', 'Vz')
    Vxi = load_2d_moment_data(timestep, 'ion', 'Vx')
    Vyi = load_2d_moment_data(timestep, 'ion', 'Vy')
    Vzi = load_2d_moment_data(timestep, 'ion', 'Vz')
    
    ne_count = load_2d_moment_data(timestep, 'electron', 'density_count')
    ni_count = load_2d_moment_data(timestep, 'ion', 'density_count')
    
    # --- 2. 派生量の計算 ---
    print("派生量を計算中...")
    
    Psi = calculate_magnetic_flux(Bx, By, DELX)
    Te_proxy = Vxe**2 + Vye**2 + Vze**2
    Ti_proxy = Vxi**2 + Vyi**2 + Vzi**2
    
    J_data = {'density_count_e': ne_count, 'density_count_i': ni_count,
              'Vx_e': Vxe, 'Vx_i': Vxi, 'Vy_e': Vye, 'Vy_i': Vyi,
              'Vz_e': Vze, 'Vz_i': Vzi}
              
    Jx, Jy, Jz, _ = calculate_current_density(Bx, By, Ex, Ey, Ez, J_data, 1.0)
    
    Ez_non_ideal_electron = Ez + (Vxe * By - Vye * Bx)
    Ez_non_ideal_ion = Ez + (Vxi * By - Vyi * Bx)
    
    # --- 3. 座標グリッドの作成 ---
    X, Y = create_coordinates(GLOBAL_NX_PHYS, GLOBAL_NY_PHYS)

    # --- 4. 可視化実行 ---
    
    # (a) 個別プロット用のリスト
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
        ('Vxi', Vxi, 'Ion Velocity (Vx)', '$V_{ix}$', plt.cm.RdBu_r),
        ('Vyi', Vyi, 'Ion Velocity (Vy)', '$V_{iy}$', plt.cm.RdBu_r),
        ('Vzi', Vzi, 'Ion Velocity (Vz)', '$V_{iz}$', plt.cm.RdBu_r),
        ('Vxe', Vxe, 'Electron Velocity (Vx)', '$V_{ex}$', plt.cm.RdBu_r),
        ('Vye', Vye, 'Electron Velocity (Vy)', '$V_{ey}$', plt.cm.RdBu_r),
        ('Vze', Vze, 'Electron Velocity (Vz)', '$V_{ez}$', plt.cm.RdBu_r),
        ('Ez_non_ideal_e', Ez_non_ideal_electron, 'Non-Ideal $E_z$ (Electron)', '$E_z + (\\mathbf{V}_e \\times \\mathbf{B})_z$', plt.cm.jet),
        ('Ez_non_ideal_i', Ez_non_ideal_ion, 'Non-Ideal $E_z$ (Ion)', '$E_z + (\\mathbf{V}_i \\times \\mathbf{B})_z$', plt.cm.jet),
    ]

    # --- プロット A: 各成分を個別のサブディレクトリに出力 ---
    print("個別のプロットを生成中 (サブディレクトリに保存)...")
    
    for tag, Z, title, label, cmap in plot_components:
        SUB_DIR = os.path.join(OUTPUT_DIR, tag.replace('/', '_'))
        os.makedirs(SUB_DIR, exist_ok=True)
        vmin, vmax = get_plot_range(Z)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_single_panel(ax, X, Y, Z, Bx, By, f"Timestep {timestep}: {title}", label, cmap=cmap, 
                          vmin=vmin, vmax=vmax)
        
        fig.tight_layout()
        output_filename = os.path.join(SUB_DIR, f'plot_{timestep}_{tag}.png')
        plt.savefig(output_filename, dpi=200)
        plt.close(fig) # ★ メモリリーク対策
    
    print(f"-> 個別プロット {len(plot_components)} 点を {OUTPUT_DIR} 配下に保存しました。")

    # --- プロット B: 全ての重要な要素を1枚の統合パネルに出力 ---
    print("\n統合パネルを生成中 (ルートディレクトリに保存)...")
    
    fig, axes = plt.subplots(5, 4, figsize=(15, 18), sharex=True, sharey=True)
    ax_list = axes.flatten()
    
    # (b) 統合パネル用のリスト
    combined_plots = [
        (Bx, '(a) $B_x$', '$B_x/B_0$', plt.cm.RdBu_r),
        (By, '(b) $B_y$', '$B_y/B_0$', plt.cm.RdBu_r),
        (Bz, '(c) $B_z$', '$B_z/B_0$', plt.cm.RdBu_r),
        (Psi, '(d) $\Psi$', '$\Psi$', plt.cm.seismic),
        (Ex, '(e) $E_x$', '$E_x$', plt.cm.coolwarm),
        (Ey, '(f) $E_y$', '$E_y$', plt.cm.coolwarm),
        (Ez, '(g) $E_z$', '$E_z$', plt.cm.coolwarm),
        (ne_count, '(h) $n_e$', '$n_e$ Count', plt.cm.viridis),
        (Jx, '(i) $J_x$', '$J_x$', plt.cm.RdBu_r),
        (Jy, '(j) $J_y$', '$J_y$', plt.cm.RdBu_r),
        (Jz, '(k) $J_z$', '$J_z$', plt.cm.RdBu_r),
        (ni_count, '(l) $n_i$', '$n_i$ Count', plt.cm.viridis),
        (Vxi, '(m) $V_{ix}$', '$V_{ix}$', plt.cm.RdBu_r),
        (Vyi, '(n) $V_{iy}$', '$V_{iy}$', plt.cm.RdBu_r),
        (Vzi, '(o) $V_{iz}$', '$V_{iz}$', plt.cm.RdBu_r),
        (Ti_proxy, '(p) $T_i$ (Proxy)', '$T_i$', plt.cm.plasma),
        (Vxe, '(q) $V_{ex}$', '$V_{ex}$', plt.cm.RdBu_r),
        (Vye, '(r) $V_{ey}$', '$V_{ey}$', plt.cm.RdBu_r),
        (Vze, '(s) $V_{ez}$', '$V_{ez}$', plt.cm.RdBu_r),
        (Te_proxy, '(t) $T_e$ (Proxy)', '$T_e$', plt.cm.plasma),
    ]
    
    for i, (Z, title, label, cmap) in enumerate(combined_plots):
        if i < len(ax_list):
            ax = ax_list[i]
            vmin, vmax = get_plot_range(Z)
            stream_color = 'white' if cmap == plt.cm.seismic else 'gray'
            cf = plot_combined(ax, X, Y, Z, Bx, By, title, label, cmap=cmap, vmin=vmin, vmax=vmax, stream_color=stream_color)
        else:
            break

    fig.tight_layout()
    output_filename_combined = os.path.join(OUTPUT_DIR, f'plot_combined_{timestep}.png')
    plt.savefig(output_filename_combined, dpi=300)
    plt.close(fig) # ★ メモリリーク対策
    print(f"-> 全てを含む統合パネル (5x4) を {output_filename_combined} に保存しました。")
    print(f"--- タイムステップ: {timestep} の処理完了 ---")


# =======================================================
# ★★★ スクリプト実行ブロック (ループ処理) ★★★
# =======================================================
if __name__ == "__main__":
    # Matplotlibのフォント設定
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    # --- 引数の解析 (start, end, interval) ---
    
    # 引数の数をチェック (スクリプト名 + 3つの引数 = 合計4つ)
    if len(sys.argv) != 4:
        print("使用方法: python visual_fields.py [start_timestep] [end_timestep] [interval]")
        print("       (タイムステップは \"000500\" ではなく 500 のように数値で指定してください)")
        print("例: python visual_fields.py 0 14000 500")
        sys.exit(1)
        
    try:
        # 引数を整数として読み込む
        start_step = int(sys.argv[1])
        end_step = int(sys.argv[2])
        interval = int(sys.argv[3])
    except ValueError:
        print("エラー: タイムステップと間隔は整数で指定してください。")
        sys.exit(1)

    if interval <= 0:
        print("エラー: 間隔 (interval) は正の整数である必要があります。")
        sys.exit(1)
        
    if start_step > end_step:
        print("エラー: start_timestep は end_timestep 以下である必要があります。")
        sys.exit(1)

    print(f"--- ループ処理を開始します (Start: {start_step}, End: {end_step}, Interval: {interval}) ---")

    # --- ループ処理の実行 ---
    current_step = start_step
    while current_step <= end_step:
        
        # タイムステップを6桁のゼロ埋め文字列にフォーマット
        # (例: 500 -> "000500", 14000 -> "014000")
        timestep_str = f"{current_step:06d}"
        
        try:
            # メインの処理関数を呼び出し
            process_timestep(timestep_str)
            
        except Exception as e:
            # ★ 重大なエラーが発生しても、次のステップに進む
            print(f"★★ 重大なエラー: タイムステップ {timestep_str} の処理中に例外が発生しました: {e}")
            import traceback
            traceback.print_exc()
            print("     処理を続行します...")
            
        # 次のステップへ
        current_step += interval

    print(f"\n--- 全てのタイムステップの処理が完了しました ---")