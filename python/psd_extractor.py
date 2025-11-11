import numpy as np
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
    sys.exit(1)

# =======================================================
# Fortranバイナリ読み込み関数 (PSD用)
# (変更なし)
# =======================================================
def load_and_stitch_psd_binary(pattern, data_dir):
    """
    Fortranによって書かれたPSDバイナリファイルを読み込み、結合（スティッチ）します。
    """
    
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f"エラー: ファイルが見つかりません: {pattern}")
        return None
        
    print(f"{len(file_list)} 個のファイルを読み込みます: {os.path.join(data_dir, pattern.split('/')[-1])}")
    
    all_headers = {}
    
    # 電磁場ファイルと同じヘッダ構造を流用 (シミュレーション設定情報)
    header_dtype = np.dtype([
        ('it', 'i4'), ('np', 'i4'),
        ('nxgs', 'i4'), ('nxge', 'i4'), ('nygs', 'i4'), ('nyge', 'i4'),
        ('nxs', 'i4'), ('nxe', 'i4'), ('nys', 'i4'), ('nye', 'i4'),
        ('nsp', 'i4'), ('nproc', 'i4'), ('bc', 'i4'),
        ('delt', 'f8'), ('delx', 'f8'), ('c', 'f8')
    ])

    # ファイル名から速度空間のグリッドサイズを取得 (例: 0300-0100 -> NVX=300, NVY=100)
    try:
        # パターンからファイル名の一部を抽出するため、glob.globの結果からサンプルのファイル名を取得
        # ※ ここはファイルが一つも存在しない場合にエラーになる可能性があるので、
        #    より堅牢にするには、事前に引数から NVX, NVY を受け取るようにすると良いです。
        sample_file_name = pattern.split('/')[-1].replace('*', '00300-0100') # 仮のファイル名の一部
        nv_parts = sample_file_name.split('_')[-2].split('-')
        NV_X = int(nv_parts[0]) 
        NV_Y = int(nv_parts[1]) 
    except Exception:
        NV_X = 32 # デフォルト値
        NV_Y = 32 # デフォルト値
        
    print(f"  -> 速度空間グリッドサイズ (NVX={NV_X}, NVY={NV_Y}) と仮定しました。")
    
    global_nx_phys = 0
    global_ny_phys = 0

    print("  ... パス 1/2: ヘッダをスキャン中 ...")
    for f in file_list:
        try:
            ff = FortranFile(f, 'r')
            header_data = ff.read_record(dtype=header_dtype)[0]
            all_headers[f] = header_data
            
            # グローバル物理領域のグリッドサイズを決定 (Ghostセルを含まない)
            global_nx_phys = header_data['nxge'] - header_data['nxgs'] + 1 
            global_ny_phys = header_data['nyge'] - header_data['nygs'] + 1
            
            ff.close()
            
        except Exception as e:
            print(f"    エラー: {f} のヘッダ読み込みに失敗: {e}")
            return None

    if global_nx_phys == 0 or global_ny_phys == 0:
        print("エラー: グローバル物理領域グリッドサイズを決定できませんでした。")
        return None
        
    print(f"  -> グローバル物理グリッドサイズを (NX={global_nx_phys}, NY={global_ny_phys}) と決定しました。")

    # PSDデータ配列を初期化 (NV_Y, NV_X, NY_phys, NX_phys)
    global_psd = np.zeros((NV_Y, NV_X, global_ny_phys, global_nx_phys))
    
    print("  ... パス 2/2: データを読み込み・結合中 ...")
    
    # PSDデータのローカルサイズは、物理領域のセル数に依存
    nx_phys_local = (header_data['nxe'] - header_data['nxs'] + 1)
    ny_phys_local = (header_data['nye'] - header_data['nys'] + 1)
    
    # PSDは NVX * NVY * (NX_phys) * (NY_phys) のデータ
    psd_flat_size = NV_X * NV_Y * nx_phys_local * ny_phys_local
    psd_local_shape = (NV_Y, NV_X, nx_phys_local, ny_phys_local) # Fortranの並び方 (VY, VX, NX, NY) を仮定

    for f, header in all_headers.items():
        try:
            ff = FortranFile(f, 'r')
            
            # Record 1-4 (ヘッダ、np2, q, r) と Record 5 (電磁場 uf) をスキップ
            ff.read_record(dtype=header_dtype) 
            ff.read_ints('i4')   # Record 2: np2 (スキップ)
            ff.read_reals('f8')  # Record 3: q (スキップ)
            ff.read_reals('f8')  # Record 4: r (スキップ)
            
            # Record 5: 電磁場データ (uf) をスキップ
            nx_local_uf = header['nxe'] - header['nxs'] + 3
            ny_local_uf = header['nye'] - header['nys'] + 3
            uf_size = 6 * nx_local_uf * ny_local_uf
            ff.read_reals('f8', shape=uf_size)
            
            # Record 6: PSDデータ
            data_flat = ff.read_reals('f8')
            
            if data_flat.size != psd_flat_size:
                raise ValueError(f"PSDデータサイズ不一致。 期待値={psd_flat_size}, 実際={data_flat.size}")
            
            # Fortranの列優先 (order='F') で読み込み、 PSD(NVY, NVX, NX_local, NY_local) の形にする
            psd_data_local = data_flat.reshape(psd_local_shape, order='F')
            
            # グローバル配列への貼り付け位置を計算 (物理領域インデックス)
            g_start_x = header['nxs'] - header['nxgs']
            g_end_x   = g_start_x + nx_phys_local
            g_start_y = header['nys'] - header['nygs']
            g_end_y   = g_start_y + ny_phys_local

            # Python/NumPyの標準形式 (NVY, NVX, NY_phys, NX_phys) にするために転置 (0, 1, 3, 2) を行う
            global_psd[:, :, g_start_y:g_end_y, g_start_x:g_end_x] = psd_data_local.transpose(0, 1, 3, 2)

            ff.close() 

        except Exception as e:
            print(f"    エラー: {f} のデータ読み込みまたは変形に失敗: {e}")
            return None

    return global_psd, all_headers[file_list[0]]


# =======================================================
# データ抽出・計算関数 (変更なし)
# =======================================================

def get_moment_data(global_psd, header):
    """
    PSDデータから速度モーメント (Vx, Vy) を計算する。
    """
    NVY, NVX, NY_phys, NX_phys = global_psd.shape
    
    # 速度グリッドを作成 (仮のスケール)
    vx_grid = np.linspace(-1.0, 1.0, NVX)
    vy_grid = np.linspace(-1.0, 1.0, NVY)
    
    VX, VY = np.meshgrid(vx_grid, vy_grid, indexing='xy') 
    
    # 密度を計算 (PSDを速度空間で積分)
    density = np.sum(global_psd, axis=(0, 1))
    
    # VxとVyのモーメントを計算
    vx_psd_product = global_psd * VX[:, :, None, None] 
    vx_moment = np.sum(vx_psd_product, axis=(0, 1)) 
    
    vy_psd_product = global_psd * VY[:, :, None, None]
    vy_moment = np.sum(vy_psd_product, axis=(0, 1)) 
    
    # 平均速度 Vx, Vy
    density_safe = np.where(density > 1e-12, density, 1e-12)
    average_vx = vx_moment / density_safe
    average_vy = vy_moment / density_safe
    
    average_vz = np.zeros_like(average_vx) 
    
    # 中心セルでの PSD分布
    center_y = NY_phys // 2
    center_x = NX_phys // 2
    center_psd = global_psd[:, :, center_y, center_x]
    
    return density, average_vx, average_vy, average_vz, center_psd

def save_data_to_txt(data_2d, label, timestep, species, out_dir, filename):
    """
    2Dデータをテキストファイルに保存する。
    """
    output_file = os.path.join(out_dir, f'data_{timestep}_{species}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"-> {species}の {label} データを {output_file} に保存しました。")

def save_psd_2d_to_txt(data_2d, label, timestep, species, out_dir, filename):
    """
    PSDの2Dデータ (vx, vy平面) をテキストファイルに保存する。
    """
    output_file = os.path.join(out_dir, f'psd_{timestep}_{species}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"-> {species}の {label} 分布 (vx-vy平面) を {output_file} に保存しました。")

# =======================================================
# メイン処理 (変更点)
# =======================================================
def main():
    # ★★★ 変更点 1: コマンドライン引数から 3 つの引数 (start, end, step) を取得 ★★★
    if len(sys.argv) < 4:
        print("使用方法: python psd_extractor.py [開始のステップ] [終了のステップ] [間隔]")
        print("例: python psd_extractor.py 000000 014000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: すべての引数 (開始、終了、間隔) は整数である必要があります。")
        sys.exit(1)
        
    print(f"--- 処理範囲: 開始={start_step}, 終了={end_step}, 間隔={step_size} ---")
    
    # 環境に応じてこのパスを調整してください
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 出力先ディレクトリ: {OUTPUT_DIR} ---")
    
    species_list = [('e', 'electron'), ('i', 'ion')] # (ファイルサフィックス, ラベル)

    # ★★★ 変更点 2: タイムステップを反復処理するループ ★★★
    for current_step in range(start_step, end_step + step_size, step_size):
        
        # タイムステップの文字列を '000500' のようにゼロ埋め6桁でフォーマット
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        for suffix, species_label in species_list:
            
            # ファイル名パターン: {timestep}_*_psd_{suffix}.dat (速度グリッド部分はワイルドカード)
            file_pattern = os.path.join(data_dir, f'{timestep}_*_psd_{suffix}.dat') 
            
            print(f"\n--- {species_label} データを処理中 ---")

            result = load_and_stitch_psd_binary(file_pattern, data_dir)
            
            if result is None:
                print(f"警告: {species_label} のファイルが見つからないか、読み込みに失敗しました。スキップします。")
                continue
                
            global_psd, header = result

            # --- 1. PSDデータからモーメント (平均速度) を計算 ---
            density, vx_moment, vy_moment, vz_moment, center_psd_2d = get_moment_data(global_psd, header)
            
            # --- 2. 各物理量をテキストファイルに保存 ---
            save_data_to_txt(density, 'Density', timestep, species_label, OUTPUT_DIR, 'density')
            save_data_to_txt(vx_moment, 'Average Velocity (Vx)', timestep, species_label, OUTPUT_DIR, 'Vx')
            save_data_to_txt(vy_moment, 'Average Velocity (Vy)', timestep, species_label, OUTPUT_DIR, 'Vy')
            save_data_to_txt(vz_moment, 'Average Velocity (Vz)', timestep, species_label, OUTPUT_DIR, 'Vz')
            save_psd_2d_to_txt(center_psd_2d, 'PSD (Center Cell)', timestep, species_label, OUTPUT_DIR, 'psd_center')
            
            print(f"--- タイムステップ {timestep} の {species_label} データのモーメント抽出・保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()