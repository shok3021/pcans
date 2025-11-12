import numpy as np

# 保存された Vx ファイルパスに合わせて変更してください
output_file_vx = '/home/shok/pcans/python/extracted_psd_data_moments/data_014000_electron_Vx.txt' 

# データを読み込む
try:
    vx_data = np.loadtxt(output_file_vx, delimiter=',')
    
    # 配列の形状と最小値・最大値を出力
    print(f"データの形状 (Vx): {vx_data.shape}")
    print(f"最小値 (Vx): {np.min(vx_data)}")
    print(f"最大値 (Vx): {np.max(vx_data)}")
    
    # 粒子が存在する Y=96 から Y=103 の行のみをチェック
    start_row = 96
    end_row = 104 
    
    # 関連する行をプリント（見やすいように小数点以下を制限）
    print(f"\nY-Index [{start_row} - {end_row-1}] の Vx データ抜粋 (X=296から303に対応する列):")
    # Y-index 96 から 103 の行、X-index 296 から 303 の列を表示
    print(vx_data[start_row:end_row, 296:304])
    
except Exception as e:
    print(f"ファイル読み込みエラー: {e}")