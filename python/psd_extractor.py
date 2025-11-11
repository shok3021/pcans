import numpy as np
import os
import glob
import sys
import time

# =======================================================
# 設定 (Fortran const モジュールから取得した正確な値)
# =======================================================
# ★★★ Fortran const モジュールからの値を使用 ★★★
# nx=321, ny=640 (グリッド点数)
GLOBAL_NX_GRID_POINTS = 321  
GLOBAL_NY_GRID_POINTS = 640

# 物理領域のグリッド数 (セル数: Grid Points - 1)
GLOBAL_NX_PHYS = GLOBAL_NX_GRID_POINTS - 1 # 320 セル
GLOBAL_NY_PHYS = GLOBAL_NY_GRID_POINTS - 1 # 639 セル

# セル幅 delx = 1.0D0
DELX = 1.0 

# X軸のスケール (diで規格化されていると仮定, L_x/d_i = NX_PHYS * DELX)
# FortranコードのBC=-1（反射境界）の場合、X方向の中心が0であることを前提とする
X_HALF_LENGTH = (GLOBAL_NX_PHYS * DELX) / 2.0 
Y_LENGTH = GLOBAL_NY_PHYS * DELX

X_MIN = -X_HALF_LENGTH # -160.0 を仮定
X_MAX = X_HALF_LENGTH  # 160.0 を仮定
Y_MIN = 0.0            # 0.0 を仮定
Y_MAX = Y_LENGTH       # 639.0 を仮定

# 可視化用のスケール (プロット例の画像スケールに近い値を使用)
# Fortranコードのグリッドサイズは 320x640 ですが、プロット画像は -6 to 6, 0 to 25 di 程度でした。
# 実際のリコネクション領域はシミュレーション全体より小さいため、抽出ロジックには全グリッドを使用しますが、
# 可視化する際は、この設定を基にスケール調整が必要になることに注意してください。
# ここでは抽出ロジックで使うグリッド数を確定します。

print(f"--- グリッド設定 ---")
print(f"X方向物理セル数: {GLOBAL_NX_PHYS}, Y方向物理セル数: {GLOBAL_NY_PHYS}")
print(f"空間範囲: X=[{X_MIN}, {X_MAX}], Y=[{Y_MIN}, {Y_MAX}] (セル幅: {DELX})")


# =======================================================
# データ抽出・計算関数 (GLOBAL定数を更新)
# =======================================================

def calculate_moments_from_particle_list(particle_data):
    """
    粒子の生データ (X, Y, Vx, Vy, Vz) から空間グリッド上の平均速度を計算する。
    """
    
    NX = GLOBAL_NX_PHYS
    NY = GLOBAL_NY_PHYS
    
    # 空間グリッドの範囲
    x_min, x_max = X_MIN, X_MAX
    y_min, y_max = Y_MIN, Y_MAX 

    # 粒子データの各列
    X_pos = particle_data[:, 0]
    Y_pos = particle_data[:, 1]
    Vx_raw = particle_data[:, 2]
    Vy_raw = particle_data[:, 3]
    Vz_raw = particle_data[:, 4]

    # グリッドビンを計算 (空間グリッドのインデックス)
    # np.linspace(min, max, N+1) の区切りを使用
    # X軸のインデックス
    bin_x = np.digitize(X_pos, np.linspace(x_min, x_max, NX + 1)[1:-1])
    # Y軸のインデックス
    bin_y = np.digitize(Y_pos, np.linspace(y_min, y_max, NY + 1)[1:-1])

    # 空間グリッド (NY, NX) を作成
    # NumPyの行列インデックス (row=Y, col=X) に合わせる
    density = np.zeros((NY, NX))
    vx_sum = np.zeros((NY, NX))
    vy_sum = np.zeros((NY, NX))
    vz_sum = np.zeros((NY, NX))
    
    # 各粒子を対応するグリッドセルに集計
    # Fortranの Y軸インデックスが昇順 (nys:nye) であると仮定し、
    # NumPyの行インデックス (iy) が Y_pos の値の昇順に対応するように処理
    for i in range(len(X_pos)):
        ix, iy = bin_x[i], bin_y[i]
        
        # インデックスが物理領域内 (0 <= index < N) であることを確認
        if 0 <= ix < NX and 0 <= iy < NY:
            density[iy, ix] += 1
            vx_sum[iy, ix] += Vx_raw[i]
            vy_sum[iy, ix] += Vy_raw[i]
            vz_sum[iy, ix] += Vz_raw[i]

    # 平均速度を計算 (ゼロ除算回避)
    density_safe = np.where(density > 0, density, 1e-12)
    
    average_vx = vx_sum / density_safe
    average_vy = vy_sum / density_safe
    average_vz = vz_sum / density_safe
    
    return density, average_vx, average_vy, average_vz

# --- (load_text_data および save_data_to_txt 関数は変更なし) ---
def load_text_data(filepath):
    if not os.path.exists(filepath):
        return None
        
    try:
        data = np.loadtxt(filepath)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    except Exception as e:
        print(f"    エラー: {filepath} のテキスト読み込みに失敗: {e}")
        return None

def save_data_to_txt(data_2d, label, timestep, species, out_dir, filename):
    output_file = os.path.join(out_dir, f'data_{timestep}_{species}_{filename}.txt')
    np.savetxt(output_file, data_2d, fmt='%.10e', delimiter=',') 
    print(f"-> {species}の {label} データを {output_file} に保存しました。")


# =======================================================
# メイン処理
# =======================================================
def main():
    if len(sys.argv) < 4:
        print("使用方法: python psd_extractor_revised.py [開始のステップ] [終了のステップ] [間隔]")
        print("例: python psd_extractor_revised.py 000000 014000 500")
        sys.exit(1)
        
    try:
        start_step = int(sys.argv[1])
        end_step   = int(sys.argv[2])
        step_size  = int(sys.argv[3])
    except ValueError:
        print("エラー: すべての引数 (開始、終了、間隔) は整数である必要があります。")
        sys.exit(1)
        
    print(f"--- 処理範囲: 開始={start_step}, 終了={end_step}, 間隔={step_size} ---")
    
    # Fortran const モジュールから読み取った値を使用
    data_dir = os.path.join('/home/shok/pcans/em2d_mpi/md_mrx/psd/')
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'extracted_psd_data_moments') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 出力先ディレクトリ: {OUTPUT_DIR} ---")
    
    species_list = [('e', 'electron'), ('i', 'ion')] 

    for current_step in range(start_step, end_step + step_size, step_size):
        
        timestep = f"{current_step:06d}" 
        print(f"\n=======================================================")
        print(f"--- ターゲットタイムステップ: {timestep} の処理を開始 ---")
        print(f"=======================================================")

        for suffix, species_label in species_list:
            
            # PSDファイルはテキスト形式で、ファイル名には速度グリッド情報が含まれていることが判明
            # ファイル名: {timestep}_0300-0100_psd_{suffix}.dat (例: 000500_0300-0100_psd_e.dat)
            filename = f'{timestep}_0300-0100_psd_{suffix}.dat'
            filepath = os.path.join(data_dir, filename)
            
            print(f"\n--- {species_label} データ ({filename}) を処理中 ---")

            particle_data = load_text_data(filepath)
            
            if particle_data is None or particle_data.size == 0:
                print(f"警告: {species_label} の粒子データが見つからないか、空です。スキップします。")
                continue
            
            print(f"  -> {len(particle_data)} 個の粒子を読み込みました。モーメントを計算中...")

            # --- 1. 粒子データからモーメント (平均速度) を計算し、空間グリッドにマップ ---
            density, average_vx, average_vy, average_vz = calculate_moments_from_particle_list(particle_data)
            
            # --- 2. 各物理量をテキストファイルに保存 ---
            save_data_to_txt(density, 'Particle Count (Density Proxy)', 
                             timestep, species_label, OUTPUT_DIR, 'density_count')
            save_data_to_txt(average_vx, 'Average Velocity (Vx)', 
                             timestep, species_label, OUTPUT_DIR, 'Vx')
            save_data_to_txt(average_vy, 'Average Velocity (Vy)', 
                             timestep, species_label, OUTPUT_DIR, 'Vy')
            save_data_to_txt(average_vz, 'Average Velocity (Vz)', 
                             timestep, species_label, OUTPUT_DIR, 'Vz')
            
            print(f"--- タイムステップ {timestep} の {species_label} データ抽出・保存が完了しました ---")

    print("\n=======================================================")
    print("=== 全ての指定されたタイムステップの処理が完了しました ===")
    print("=======================================================")


# --- スクリプトとして実行された場合にmain()を呼び出す ---
if __name__ == "__main__":
    main()