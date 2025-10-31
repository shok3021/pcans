import numpy as np
import matplotlib.pyplot as plt
import os

# --- このスクリプト (create_psd_plot.py) がある場所を基準にする ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 必要なモジュールをインポート ---
# (同じディレクトリにある前提)
try:
    from file_read import file_read
    from psd_calc import psd_calc
except ImportError:
    print(f"エラー: 'file_read.py' または 'psd_calc.py' が見つかりません。")
    print(f"このスクリプトは {SCRIPT_DIR} に配置してください。")
    exit()

# --- Matplotlibのデフォルト設定 (init.pro の代替) ---
plt.rcParams.update({
    'font.family': 'Times New Roman', # 環境にない場合は 'serif' に変更
    'font.size': 18,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# =======================================================
# メイン処理
# =======================================================
def main():
    """
    指定されたデータファイルからPSD (vx-vy 空間) を計算しプロットします。
    """
    
    # --- 1. パスの設定 ---
    # データディレクトリへの相対パス
    data_dir = os.path.join('/Users/shohgookazaki/Documents/GitHub/pcans//em2d_mpi/md_mrx/psd/')
    
    # 処理するファイル名
    filename_e = '000000_0300-0100_psd_e.dat'
    filename_i = '000000_0300-0100_psd_i.dat'
    
    # --- 2. 電子 (electron) のPSDを処理 ---
    filepath_e = os.path.join(data_dir, filename_e)
    print(f"Reading (electron): {filepath_e}")
    
    data_e = file_read(filepath_e, silent=True)
    
    if data_e is not None and data_e.ndim == 2 and data_e.shape[1] == 5:
        # 3列目 (インデックス 2) を vx
        # 4列目 (インデックス 3) を vy として抽出
        vx_e = data_e[:, 2]
        vy_e = data_e[:, 3]
        
        print(f"  ... {len(vx_e)} 個の電子データを読み込みました。")
        
        # PSDを計算
        # ビン数 (nbin) や範囲 (min/max) はデータに合わせて調整してください
        v_range = (-0.5, 0.5) # 仮の速度範囲 (例: -0.5c から 0.5c)
        n_bins = 100
        
        psd_e, vx_ax, vy_ax = psd_calc(
            vx_e, vy_e, 
            nbin_x=n_bins, nbin_y=n_bins,
            min_x=v_range[0], max_x=v_range[1],
            min_y=v_range[0], max_y=v_range[1]
        )
        
        # --- 3. プロット ---
        # psd_calc は (nx, ny) で返すため、imshow用に転置 (.T)
        # Logスケールでプロット (0を避けるため +1e-10)
        plot_data_e = np.log10(psd_e.T + 1e-10)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(
            plot_data_e, 
            origin='lower', 
            aspect='equal', # vx と vy のスケールを合わせる
            cmap='jet',
            extent=[vx_ax.min(), vx_ax.max(), vy_ax.min(), vy_ax.max()]
        )
        
        plt.colorbar(label='Log10(Counts)')
        plt.xlabel('$v_x / c$') # 軸ラベル (適宜変更)
        plt.ylabel('$v_y / c$') # 軸ラベル (適宜変更)
        plt.title(f'Electron PSD (vx-vy)\n{filename_e}')
        
        # 画像ファイルとして保存 (スクリプトと同じ場所に保存)
        output_filename_e = 'psd_plot_electron.png'
        output_path_e = os.path.join(SCRIPT_DIR, output_filename_e)
        plt.savefig(output_path_e)
        plt.close()
        
        print(f"-> 電子のPSDプロットを {output_path_e} に保存しました。")

    else:
        print(f"エラー: {filepath_e} の読み込みに失敗したか、形式が (N, 5) ではありません。")

    # (オプション) イオン (ion) も同様に処理
    # ... ここに filename_i を使った同様の処理を追加 ...
    filepath_i = os.path.join(data_dir, filename_i)
    print(f"Reading (ion): {filepath_i}")

    data_i = file_read(filepath_i, silent=True)

    if data_i is not None and data_i.ndim == 2 and data_i.shape[1] == 5:
        # 3列目 (インデックス 2) を vx
        # 4列目 (インデックス 3) を vy として抽出
        vx_i = data_i[:, 2]
        vy_i = data_i[:, 3]

        print(f"  ... {len(vx_i)} 個のイオンデータを読み込みました。")

        # PSDを計算
        v_range = (-0.5, 0.5) # 仮の速度範囲 (例: -0.5c から 0.5c)
        n_bins = 100

        psd_i, vx_ax, vy_ax = psd_calc(
            vx_i, vy_i,
            nbin_x=n_bins, nbin_y=n_bins,
            min_x=v_range[0], max_x=v_range[1],
            min_y=v_range[0], max_y=v_range[1]
        )

        # --- 3. プロット ---
        plot_data_i = np.log10(psd_i.T + 1e-10)

        plt.figure(figsize=(10, 8))
        plt.imshow(
            plot_data_i,
            origin='lower',
            aspect='equal',
            cmap='jet',
            extent=[vx_ax.min(), vx_ax.max(), vy_ax.min(), vy_ax.max()]
        )

        plt.colorbar(label='Log10(Counts)')
        plt.xlabel('$v_x / c$')
        plt.ylabel('$v_y / c$')
        plt.title(f'Ion PSD (vx-vy)\n{filename_i}')

        # 画像ファイルとして保存
        output_filename_i = 'psd_plot_ion.png'
        output_path_i = os.path.join(SCRIPT_DIR, output_filename_i)
        plt.savefig(output_path_i)
        plt.close()

        print(f"-> イオンのPSDプロットを {output_path_i} に保存しました。")

    else:
        print(f"エラー: {filepath_i} の読み込みに失敗したか、形式が (N, 5) ではありません。")

# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()