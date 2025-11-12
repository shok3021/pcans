import numpy as np

# 保存されたファイルパスに合わせて変更してください
output_file = '/home/shok/pcans/python/extracted_psd_data_moments/data_014000_electron_density_count.txt' 

# データを読み込む
try:
    density_data = np.loadtxt(output_file, delimiter=',')
    
    # 配列の形状と最小値・最大値を出力
    print(f"データの形状: {density_data.shape}")
    print(f"最小値: {np.min(density_data)}")
    print(f"最大値: {np.max(density_data)}")
    
    # 粒子が存在する Y=96 から Y=103 の行のみをチェック (NY=639 のうちの一部)
    # NumPyのインデックスは0から始まるため
    start_row = 96
    end_row = 104 
    
    # 関連する行をプリント（見やすいように小数点以下を制限）
    print(f"\nY-Index [{start_row} - {end_row-1}] のデータ抜粋 (X=296から303に対応する列):")
    # Y-index 96 から 103 の行、X-index 296 から 303 の列を表示
    print(density_data[start_row:end_row, 296:304])
    
except Exception as e:
    print(f"ファイル読み込みエラー: {e}")