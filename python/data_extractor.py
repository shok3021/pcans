import numpy as np
import os
import glob
import sys
import time
from scipy.io import FortranFile

# =======================================================
# Fortranバイナリ読み込み関数 (粒子データ抽出ロジックを追加)
# =======================================================
def load_and_stitch_fortran_binary(pattern, output_dir_particles, timestep):
    """
    Fortranバイナリファイルを読み込み、フィールドを結合し、粒子生データを保存する。
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
        
    # フィールドは 6成分 (Bx, By, Bz, Ex, Ey, Ez)
    global_fields = np.zeros((6, global_ny_full, global_nx_full))
    
    # Fortranヘッダを代表ファイルから取得
    representative_header = all_headers[file_list[0]]
    NP_max = representative_header['np']
    NSP = representative_header['nsp']
    
    print("  ... パス 2/2: データを読み込み・結合中 ...")
    
    for rank, (f, header) in enumerate(all_headers.items()):
        try:
            ff = FortranFile(f, 'r')
            
            # --- 1. ヘッダレコード (スキップ) ---
            ff.read_record(dtype=header_dtype) 
            
            # --- 2. np2 (粒子数マップ) ---
            NY_local = header['nye'] - header['nys'] + 1
            # np2 は (NY_local, NSP) の形状で書かれている
            np2_flat = ff.read_ints('i4')
            np2_local = np2_flat.reshape((NY_local, NSP), order='F')

            # --- 3. q (電荷) ---
            ff.read_reals('f8') 
            
            # --- 4. r (質量) ---
            ff.read_reals('f8') 

            # --- 5. uf (フィールドデータ) ---
            nx_local_written = header['nxe'] - header['nxs'] + 3
            ny_local_written = header['nye'] - header['nys'] + 3
            data_flat = ff.read_reals('f8')
            
            field_data_local = data_flat.reshape((6, nx_local_written, ny_local_written), order='F')
            
            # グローバル配列への貼り付け (フィールド)
            g_start_x = header['nxs'] - header['nxgs']
            g_end_x   = g_start_x + nx_local_written
            g_start_y = header['nys'] - header['nygs']
            g_end_y   = g_start_y + ny_local_written
            global_fields[:, g_start_y:g_end_y, g_start_x:g_end_x] = field_data_local.transpose(0, 2, 1)

            # --- 6. up (粒子データ) ---
            # up は (5, NP_max, NY_local, NSP) の形状で書かれている
            # NP_max (200 * NX), NY_local (nys:nye のサイズ), NSP (2)
            particles_flat = ff.read_reals('f8')
            
            # (5, NP, NY_local, NSP) の形状に戻す
            expected_size = 5 * NP_max * NY_local * NSP
            if particles_flat.size != expected_size:
                raise ValueError(f"Particle (up): データサイズ不一致。 期待値={expected_size}, 実際={particles_flat.size}")
                
            particles_local = particles_flat.reshape((5, NP_max, NY_local, NSP), order='F')
            
            # --- 粒子データの保存 ---
            for isp in range(NSP):
                species_label = 'ion' if isp == 0 else 'electron'
                output_file = os.path.join(output_dir_particles, 
                                           f'raw_{timestep}_{species_label}_rank_{rank}.txt')
                
                # 実際に存在する粒子を抽出して保存 (不要なゼロパディングは除く)
                # (X, Y, Vx, Vy, Vz) の順に結合し、粒子数がゼロでない行のみを選択
                all_particles_list = []
                for j in range(NY_local):
                    # Fortranの nys:nye の範囲を NumPy の 0:NY_local-1 にマッピング
                    y_idx = j
                    # 実際に存在する粒子数
                    n_actual = np2_local[y_idx, isp] 
                    
                    if n_actual > 0:
                        # (5, NP_max) -> (NP_max, 5) の形状に転置し、n_actual 行を抽出
                        current_particles = particles_local[:, :n_actual, y_idx, isp].transpose(1, 0)
                        all_particles_list.append(current_particles)
                
                if all_particles_list:
                    combined_particles = np.concatenate(all_particles_list, axis=0)
                    np.savetxt(output_file, combined_particles, fmt='%.10e', delimiter=',')
                    print(f"  -> Rank {rank}, {species_label}: {len(combined_particles)} 粒子の生データを保存しました。")
                else:
                    print(f"  -> Rank {rank}, {species_label}: 粒子データなし。")


            ff.close() 

        except Exception as e:
            print(f"    エラー: {f} のデータ読み込みまたは変形に失敗: {e}")
            return None

    print("  ... 全ファイルの処理が完了しました。")
    return global_fields, representative_header


# ... (get_physical_region, save_data_to_txt の定義は変更なし) ...

def get_physical_region(global_fields, header):
    """Ghostセルを除いた物理領域を切り出す (規格化された値)"""
    # Fortranの const モジュールから nx=321, ny=640 と仮定
    NX_CELLS_PHYS = (header['nxge'] - header['nxgs'] + 1) - 1 # 321 - 1 = 320
    NY_CELLS_PHYS = (header['nyge'] - header['nygs'] + 1) - 1 # 640 - 1 = 639
    
    # 物理領域は Ghost セルの次のインデックスから始まる [1:NY_CELLS_PHYS+1, 1:NX_CELLS_PHYS+1]
    phys_start = 1
    phys_end_x = phys_start + NX_CELLS_PHYS
    phys_end_y = phys_start + NY_CELLS_PHYS
    
    # (6, NY_phys, NX_phys) の配列を返す (サイズ 6, 639, 320)
    return global_fields[:, phys_start:phys_end_y, phys_start:phys_end_x]

def save_data_to_txt(data_2d, label, timestep, out_dir, filename):
    """
    2Dデータをテキストファイルに保存する。
    """
    output_file = os.path.join(out_dir, f'data_{timestep}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"-> 規格化された {label} データを {output_file} に保存しました。")

# =======================================================
# メイン処理
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python data_extractor.py [開始のステップ] [終了のステップ] [間隔]")
        print("例: python data_extractor.py 000000 014000 500")
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
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/dat/')
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR_FIELDS = os.path.join(SCRIPT_DIR, 'extracted_data') 
    OUTPUT_DIR_PARTICLES = os.path.join(SCRIPT_DIR, 'raw_particle_data') # 新しいディレクトリ
    
    os.makedirs(OUTPUT_DIR_FIELDS, exist_ok=True)
    os.makedirs(OUTPUT_DIR_PARTICLES, exist_ok=True)
    
    print(f"--- 出力先ディレクトリ (場): {OUTPUT_DIR_FIELDS} ---")
    print(f"--- 出力先ディレクトリ (粒子): {OUTPUT_DIR_PARTICLES} ---")
    
    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        file_pattern = os.path.join(data_dir, f'{timestep}_rank=*.dat') 
        
        # --- Fortranバイナリの読み込みと結合 ---
        start_time = time.time()
        # 粒子データの出力ディレクトリを渡す
        result = load_and_stitch_fortran_binary(file_pattern, OUTPUT_DIR_PARTICLES, timestep)
        end_time = time.time()
        
        if result is None:
            print(f"警告: タイムステップ {timestep} のファイルが見つからないか、読み込みに失敗しました。スキップします。")
            continue
            
        global_fields, header = result
        print(f"  -> 処理時間 (読み込み/結合/粒子保存): {end_time - start_time:.2f} 秒")

        # --- 1. 物理領域を切り出す (場データ) ---
        phys_fields = get_physical_region(global_fields, header)
        
        # 各成分を切り出し
        bx_phys = phys_fields[0, :, :]
        by_phys = phys_fields[1, :, :]
        bz_phys = phys_fields[2, :, :]
        ex_phys = phys_fields[3, :, :]
        ey_phys = phys_fields[4, :, :]
        ez_phys = phys_fields[5, :, :]

        # --- 2. 各物理量をテキストファイルに保存 ---
        save_data_to_txt(bx_phys, 'Magnetic Field (Bx)', timestep, OUTPUT_DIR_FIELDS, 'Bx')
        save_data_to_txt(by_phys, 'Magnetic Field (By)', timestep, OUTPUT_DIR_FIELDS, 'By')
        save_data_to_txt(bz_phys, 'Magnetic Field (Bz)', timestep, OUTPUT_DIR_FIELDS, 'Bz')
        save_data_to_txt(ex_phys, 'Electric Field (Ex)', timestep, OUTPUT_DIR_FIELDS, 'Ex')
        save_data_to_txt(ey_phys, 'Electric Field (Ey)', timestep, OUTPUT_DIR_FIELDS, 'Ey')
        save_data_to_txt(ez_phys, 'Electric Field (Ez)', timestep, OUTPUT_DIR_FIELDS, 'Ez')
        
        print(f"--- タイムステップ {timestep} の処理が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()