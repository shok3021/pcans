import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import griddata

# =======================================================
# 設定
# =======================================================
# ★★★ データが格納されているディレクトリとファイル名の形式を調整してください ★★★
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'extracted_data') 

# 物理量に合わせてスケールを調整してください。
# 例: x軸とy軸のスケール (diで規格化されていると仮定)
X_SCALE = 6.5
Y_SCALE = 25.0

# =======================================================
# ヘルパー関数
# =======================================================

def load_data(timestep, component):
    """
    指定されたタイムステップと成分のテキストファイル (CSV) を読み込む。
    """
    filename = f'data_{timestep}_{component}.txt'
    filepath = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return None
        
    # numpy.loadtxt でCSV (delimiter=',') 形式のデータを読み込む
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception as e:
        print(f"エラー: {filepath} の読み込み中にエラーが発生しました: {e}")
        return None

def create_coordinates(data):
    """
    データの形状から、プロット用の X, Y 座標配列を作成する。
    """
    NY, NX = data.shape
    
    # x/di は -X_SCALE から X_SCALE まで、y/di は 0 から Y_SCALE までと仮定
    x = np.linspace(-X_SCALE, X_SCALE, NX)
    y = np.linspace(0, Y_SCALE, NY)
    
    # メッシュグリッドを作成
    X, Y = np.meshgrid(x, y)
    return X, Y

# =======================================================
# プロット関数
# =======================================================

def plot_field(ax, X, Y, Z, title, vmin, vmax, cmap='RdBu_r', plot_streamlines=False, Bx=None, By=None):
    """
    ContourfとStreamlinesをプロットする。
    """
    # Contours (カラーマップ)
    # vminとvmaxを固定することで、パネル間で色のスケールを合わせる
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, format='%.2f')
    
    # Streamlines (磁力線または流線)
    if plot_streamlines and Bx is not None and By is not None:
        # 密度を調整してプロット
        ax.streamplot(X[::2, ::2], Y[::2, ::2], 
                      Bx[::2, ::2], By[::2, ::2], 
                      color='gray', linewidth=0.5, density=1.0, 
                      arrowstyle='-', minlength=0.1, zorder=1)
    
    # ラベルとタイトル
    ax.set_xlabel('$x/d_i$')
    ax.set_ylabel('$y/d_i$')
    ax.set_title(title)
    
    # 軸の範囲を設定 (実際のデータ範囲と合わせる)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    
    # グリッド
    ax.tick_params(direction='in', top=True, right=True)


# =======================================================
# メイン処理
# =======================================================
def main():
    # ★★★ コマンドライン引数から timestep を取得 ★★★
    if len(sys.argv) < 2:
        print("使用方法: python visualize_fields.py [timestep]")
        print("例: python visualize_fields.py 000500")
        sys.exit(1)
        
    timestep = sys.argv[1] # 最初の引数をタイムステップとして取得
    print(f"--- ターゲットタイムステップ: {timestep} ---")
    
    # --- データの読み込み ---
    print("データを読み込み中...")
    
    # 電磁場データ (Bx, By は磁力線/流線用)
    Bx = load_data(timestep, 'Bx')
    By = load_data(timestep, 'By')
    Bz = load_data(timestep, 'Bz')
    
    # 速度データ (ここでは簡単のため Bx, By, Bz, Ex, Ey, Ez のみを使用し、
    # Vz は計算が複雑になるため、画像(d)の再現にはEz, Bx, Byを使用)
    Ex = load_data(timestep, 'Ex')
    Ey = load_data(timestep, 'Ey')
    Ez = load_data(timestep, 'Ez')
    
    if Bx is None or By is None or Bz is None or Ez is None:
        print("必要なデータの一部またはすべてが読み込めませんでした。処理を終了します。")
        return

    # --- 座標の作成 ---
    X, Y = create_coordinates(Bx)
    
    # --- 派生量 (Panel d) の計算 ---
    # Ez + (V_e x B)_z / c を計算するには、電子の速度 V_e が必要です。
    # 元のスクリプトには電磁場しかないので、ここでは電磁場から直接計算できる量
    # (例: Ez のみ、または B_mag) をプロット対象とします。
    # 元の画像 (d) は $E_z + (\mathbf{V}_e \times \mathbf{B})_z / c$ をプロットしている可能性が高いため、
    # データの不足を補うために、ここでは **Ez** をプロット対象として代用します。
    # または、リコネクションで重要な量である **Bz** をプロットします。
    # (ここでは一旦、元の画像の雰囲気から Ez を代用します)
    Derived_Field = Ez # ここを実際のプロットしたい物理量に変更してください
    
    
    # --- プロットの実行 ---
    print("プロットを開始します...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    ax_list = axes.flatten()
    
    # パネル (a): Vx (データがないため By を代用。実際はVxが必要です)
    plot_field(ax_list[0], X, Y, By, 
               title='(a) $V_x / V_{A0}$ (Proxy: $B_y / B_0$)', 
               vmin=-1.5, vmax=1.5, plot_streamlines=True, Bx=Bx, By=By) 

    # パネル (b): Vy (データがないため Bx を代用。実際はVyが必要です)
    plot_field(ax_list[1], X, Y, Bx, 
               title='(b) $V_y / V_{A0}$ (Proxy: $B_x / B_0$)', 
               vmin=-1.5, vmax=1.5, plot_streamlines=True, Bx=Bx, By=By) 

    # パネル (c): Bz
    plot_field(ax_list[2], X, Y, Bz, 
               title='(c) $B_z / B_0$', 
               vmin=-0.15, vmax=0.15, plot_streamlines=True, Bx=Bx, By=By) 

    # パネル (d): Ez (Ez + V_e x B の代用)
    # Ez のみに Streamlines をプロットしても意味がないため、Contourfのみ
    plot_field(ax_list[3], X, Y, Derived_Field, 
               title='(d) $(E_z + (\mathbf{V} \\times \\mathbf{B})_z / c) / B_0$ (Proxy: $E_z / B_0$)', 
               vmin=-0.08, vmax=0.08, cmap='jet') 

    # レイアウト調整と保存
    fig.tight_layout()
    output_filename = os.path.join(DATA_DIR, f'visualization_{timestep}.png')
    plt.savefig(output_filename, dpi=300)
    print(f"\n--- 可視化結果を {output_filename} に保存しました ---")
    plt.show()

# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    # Matplotlibのフォント設定をLaTeX風にするとより画像に近くなります
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'
    
    main()