import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import time

# --- Scipy をインポート ---
try:
    from scipy.io import FortranFile
except ImportError:
    print("エラー: 'scipy' ライブラリが見つかりません。")
    print("ターミナルで 'pip install scipy' を実行してください。")
    sys.exit()

# --- 必要なモジュールをインポート ---
try:
    from field_lines_2d import field_lines_2d
except ImportError:
    print(f"エラー: 'field_lines_2d.py' が見つかりません。")
    print(f"このスクリプトと同じディレクトリに配置してください。")
    sys.exit()

# --- Matplotlibのデフォルト設定 (省略) ---
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 18, 'axes.linewidth': 2,
    'xtick.major.width': 2, 'ytick.major.width': 2, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
})

# =======================================================
# Fortranバイナリ読み込み関数 (変更なし)
# =======================================================
def load_and_stitch_fortran_binary(pattern):
    """
    fio__output (Fortran) によって書かれたバイナリファイルを読み込み、
    電磁場データ (uf) を結合（スティッチ）します。
    """
    
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f"エラー: ファイルが見つかりません: {pattern}")
        return None
        
    print(f"{len(file_list)} 個のファイルを読み込みます: {pattern}")
    
    all_headers = {}
    global_nx_full = 0
    global_ny_full = 0
    
    header_dtype = np.dtype([
        ('it', 'i4'), ('np', 'i4'),
        ('nxgs', 'i4'), ('nxge', 'i4'), ('nygs', 'i4'), ('nyge', 'i4'),
        ('nxs', 'i4'), ('nxe', 'i4'), ('nys', 'i4'), ('nye', 'i4'),
        ('nsp', 'i4'), ('nproc', 'i4'), ('bc', 'i4'),
        ('delt', 'f8'), ('delx', 'f8'), ('c', 'f8')
    ])

    print("  ... パス 1/2: ヘッダをスキャン中 ...")
    for f in file_list:
        try:
            ff = FortranFile(f, 'r')
            header_data = ff.read_record(dtype=header_dtype)[0]
            all_headers[f] = header_data
            
            global_nx_full = header_data['nxge'] - header_data['nxgs'] + 1 + 2 
            global_ny_full = header_data['nyge'] - header_data['nygs'] + 1 + 2
            
            ff.close()
            
        except Exception as e:
            print(f"    エラー: {f} のヘッダ読み込みに失敗: {e}")
            return None

    if global_nx_full == 0 or global_ny_full == 0:
        print("エラー: グローバルグリッドサイズを決定できませんでした。")
        return None
        
    print(f"  -> グローバルグリッドサイズ (Ghost含む) を (NX={global_nx_full}, NY={global_ny_full}) と決定しました。")

    global_fields = np.zeros((6, global_ny_full, global_nx_full))
    
    print("  ... パス 2/2: データを読み込み・結合中 ...")
    
    for f, header in all_headers.items():
        # print(f"  ... 処理中: {os.path.basename(f)}") # 詳細すぎるのでコメントアウト
        try:
            ff = FortranFile(f, 'r')
            
            ff.read_record(dtype=header_dtype) 
            ff.read_ints('i4')   # Record 2: np2 (スキップ)
            ff.read_reals('f8')  # Record 3: q (スキップ)
            ff.read_reals('f8')  # Record 4: r (スキップ)

            nx_local_written = header['nxe'] - header['nxs'] + 3
            ny_local_written = header['nye'] - header['nys'] + 3
            data_flat = ff.read_reals('f8')
            
            if data_flat.size != (6 * nx_local_written * ny_local_written):
                raise ValueError(f"Field (uf): データサイズ不一致。 期待値={6 * nx_local_written * ny_local_written}, 実際={data_flat.size}")
            
            field_data_local = data_flat.reshape((6, nx_local_written, ny_local_written), order='F')
            
            g_start_x = header['nxs'] - header['nxgs']
            g_end_x   = g_start_x + nx_local_written
            g_start_y = header['nys'] - header['nygs']
            g_end_y   = g_start_y + ny_local_written

            global_fields[:, g_start_y:g_end_y, g_start_x:g_end_x] = field_data_local.transpose(0, 2, 1)

            ff.close() 

        except Exception as e:
            print(f"    エラー: {f} のデータ読み込みまたは変形に失敗: {e}")
            return None

    print("  ... 全ファイルの結合が完了しました。")
    return global_fields, all_headers[file_list[0]]


# =======================================================
# プロット関数群 (ここから変更)
# =======================================================

def get_physical_region(global_fields, header):
    """Ghostセルを除いた物理領域を切り出す"""
    nxgs, nxge = header['nxgs'], header['nxge']
    nygs, nyge = header['nygs'], header['nyge']
    
    phys_start_x = nxgs - (nxgs-1)
    phys_end_x   = nxge - (nxgs-1) + 1
    phys_start_y = nygs - (nygs-1)
    phys_end_y   = nyge - (nygs-1) + 1
    
    return global_fields[:, phys_start_y:phys_end_y, phys_start_x:phys_end_x]

# ★★★ 変更点 1: 磁力線を「計算する」関数を分離 ★★★
def calculate_field_lines(bx_phys, by_phys):
    """磁力線を計算して、プロット用のデータを返す"""
    print("  ... 磁力線を計算中 ...")
    try:
        phys_ny, phys_nx = bx_phys.shape
        # npos=流線数, nsteps=ステップ数, length=1ステップ長
        field_lines = field_lines_2d(bx_phys.T, by_phys.T, npos=100, nsteps=phys_nx*2, length=0.5)
        return field_lines
    except Exception as e:
        print(f"  エラー: field_lines_2d の実行に失敗: {e}")
        return None

# ★★★ 変更点 2: 2Dマッププロット関数に「重ね書き」機能を追加 ★★★
def plot_2d_map(data_2d, title, label, cmap, timestep, out_dir, filename, field_lines_data=None):
    """
    2Dカラーマップをプロットする共通関数。
    オプションで磁力線を重ね書きする。
    """
    print(f"  ... {title} をプロット中 ...")
    phys_ny, phys_nx = data_2d.shape
    
    plt.figure(figsize=(10, 10 * phys_ny / phys_nx))
    
    vmax = np.abs(data_2d).max()
    if vmax == 0: vmax = 1.0 
    
    # 1. 背景のカラーマップを描画
    plt.imshow(data_2d, origin='lower', aspect='equal', 
               cmap=cmap, vmin=-vmax, vmax=vmax,
               extent=[0, phys_nx-1, 0, phys_ny-1])
    
    plt.colorbar(label=label)
    
    # 2. 磁力線を重ね書き (オプション)
    if field_lines_data is not None:
        print(f"    ... 磁力線を重ね書き ...")
        for k in range(field_lines_data.shape[2]): # npos
            plt.plot(field_lines_data[0, :, k], field_lines_data[1, :, k], 
                     color='black', linewidth=0.3) # 色=黒, 細線

    # 3. 軸とタイトルを設定
    plt.xlabel('X (grid)')
    plt.ylabel('Y (grid)')
    plt.title(f'{title} (Timestep {timestep})')
    plt.xlim(0, phys_nx - 1)
    plt.ylim(0, phys_ny - 1)
    
    # 4. ファイルに保存 (out_dir に)
    output_file = os.path.join(out_dir, f'plot_{timestep}_{filename}.png')
    plt.savefig(output_file)
    plt.close()
    print(f"-> {title} プロットを {output_file} に保存しました。")

# =======================================================
# メイン処理 (変更)
# =======================================================
def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join('/Users/shohgookazaki/Documents/Github/pcans/em2d_mpi/md_mrx/dat/')
    
    # ★★★ 変更点 3: 出力ディレクトリを作成 ★★★
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dat')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 出力先ディレクトリ: {OUTPUT_DIR} ---")
    
    # ★★★ ここを編集してください ★★★
    timestep = "000000" # 例: "000000" から "003000" に変更
    # ★★★★★★★★★★★★★★★★★
    
    file_pattern = os.path.join(data_dir, f'{timestep}_rank=*.dat') 
    
    result = load_and_stitch_fortran_binary(file_pattern)
    
    if result is None:
        print("処理を終了します。")
        return
        
    global_fields, header = result

    # --- 1. 物理領域を切り出す ---
    phys_fields = get_physical_region(global_fields, header)
    
    bx_phys = phys_fields[0, :, :]
    by_phys = phys_fields[1, :, :]
    bz_phys = phys_fields[2, :, :]
    ex_phys = phys_fields[3, :, :]
    ey_phys = phys_fields[4, :, :]
    ez_phys = phys_fields[5, :, :]

    # --- 2. 磁力線を「一度だけ」計算 ---
    field_lines = calculate_field_lines(bx_phys, by_phys)

    # --- 3. 各物理量をプロット ---
    
    # (a) Bx + 磁力線
    plot_2d_map(bx_phys, 'Magnetic Field (Bx)', '$B_x$', 'RdBu_r', 
                timestep, OUTPUT_DIR, 'Bx', field_lines_data=field_lines)

    # (b) By + 磁力線
    plot_2d_map(by_phys, 'Magnetic Field (By)', '$B_y$', 'RdBu_r', 
                timestep, OUTPUT_DIR, 'By', field_lines_data=field_lines)
    
    # (c) Bz + 磁力線
    plot_2d_map(bz_phys, 'Magnetic Field (Bz)', '$B_z$', 'RdBu_r', 
                timestep, OUTPUT_DIR, 'Bz', field_lines_data=field_lines)

    # (d) Ex + 磁力線
    plot_2d_map(ex_phys, 'Electric Field (Ex)', '$E_x$', 'RdBu_r', 
                timestep, OUTPUT_DIR, 'Ex', field_lines_data=field_lines)

    # (e) Ey + act
    plot_2d_map(ey_phys, 'Electric Field (Ey)', '$E_y$', 'RdBu_r', 
                timestep, OUTPUT_DIR, 'Ey', field_lines_data=field_lines)
    
    # (f) Ez + 磁力線
    plot_2d_map(ez_phys, 'Electric Field (Ez)', '$E_z$', 'RdBu_r', 
                timestep, OUTPUT_DIR, 'Ez', field_lines_data=field_lines)
    
    print("\n--- 電磁場6成分のプロット完了 ---")
    
    print("\n注: イオン/電子の密度・速度、磁束(PSI)は、")
    print("   このスクリプトではまだ読み込まれていません。")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()