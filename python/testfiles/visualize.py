import numpy as np
import matplotlib.pyplot as plt
import os

# --- 変換した .py ファイル（モジュール）をインポート ---
# (init.pyは設定ファイルなので、ここでは直接設定を記述します)
from file_read import file_read
from psd_calc import psd_calc
from fftf import fftf
from fftf2d import fftf2d
from field_lines_2d import field_lines_2d

# --- Matplotlibのデフォルト設定 (init.pro の代替) ---
plt.rcParams.update({
    'font.family': 'Times New Roman', # 環境にない場合は 'serif' などに変更
    'font.size': 18,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# --- データディレクトリのパス ---
DATA_DIR = './data/'

# =======================================================
# Tutorial 1: 1Dデータのプロット (line.dat)
# =======================================================
print("Tutorial 1: 1D Plot")
try:
    data1d = file_read(os.path.join(DATA_DIR, 'line.dat'))
    if data1d is not None and data1d.ndim == 2:
        x = data1d[:, 0]
        y = data1d[:, 1]
        
        plt.figure(figsize=(10, 8))
        plt.plot(x, y, color='blue', linewidth=2)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Tutorial 1: 1D Plot (line.dat)')
        plt.savefig('tutorial_1_line.png')
        plt.close()
        print("-> tutorial_1_line.png を保存しました。")
except Exception as e:
    print(f"Tutorial 1 失敗: {e}")

# =======================================================
# Tutorial 2: 2Dデータのプロット (image.dat)
# =======================================================
print("\nTutorial 2: 2D Image Plot")
try:
    data2d = file_read(os.path.join(DATA_DIR, 'image.dat'))
    if data2d is not None and data2d.ndim == 2:
        # IDLの (col, row) をNumpyの (row, col) に合わせるため転置(.T)
        data_to_plot = data2d.T 
        
        plt.figure(figsize=(10, 8))
        # aspect='auto' でピクセルを正方形にしない (IDLの image に近い)
        # origin='lower' で (0,0) を左下に
        plt.imshow(data_to_plot, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Value')
        plt.xlabel('X-index')
        plt.ylabel('Y-index')
        plt.title('Tutorial 2: 2D Image (image.dat)')
        plt.savefig('tutorial_2_image.png')
        plt.close()
        print("-> tutorial_2_image.png を保存しました。")
except Exception as e:
    print(f"Tutorial 2 失敗: {e}")

# =======================================================
# Tutorial 3: 2Dベクトルと流線 (vector.dat)
# =======================================================
print("\nTutorial 3: Vector Plot & Field Lines")
try:
    vec_data = file_read(os.path.join(DATA_DIR, 'vector.dat'))
    if vec_data is not None:
        # データを分離 (X, Y, Ux, Uy)
        # IDLは (col, row) だが、file_readがどう読み込むかによる
        # ここでは (N, 4) -> (N,) x 4 と想定
        if vec_data.shape[1] == 4:
            x_pos = vec_data[:, 0]
            y_pos = vec_data[:, 1]
            ux_val = vec_data[:, 2]
            uy_val = vec_data[:, 3]

            # (1) ベクトルプロット (Quiver)
            plt.figure(figsize=(10, 10))
            # データをグリッド化していないため、quiverで直接プロット
            plt.quiver(x_pos, y_pos, ux_val, uy_val, scale=50) # scaleは適宜調整
            plt.title('Tutorial 3: Vector Plot (vector.dat)')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.axis('equal')
            plt.savefig('tutorial_3_vector.png')
            plt.close()
            print("-> tutorial_3_vector.png を保存しました。")

            # (2) 流線プロット (field_lines_2d)
            # 元データがグリッドデータでないため、field_lines_2d.py を
            # 使うには、まずデータをグリッドに補間する必要があります。
            # 今回は field_lines_2d.py がグリッドデータ(ux, uy)を
            # 受け取ることを想定し、Matplotlib標準のstreamplotで代替します。
            
            # もし vector.dat が (nx, ny, 2) のグリッドデータなら
            # ux = vec_data[:, :, 0]
            # uy = vec_data[:, :, 1]
            # r = field_lines_2d(ux, uy, npos=50, nsteps=1000, length=0.1)
            # plt.figure(figsize=(10, 10))
            # for k in range(r.shape[2]): # npos
            #     plt.plot(r[0, :, k], r[1, :, k], color='black', linewidth=0.5)
            # ...
            
            # Matplotlibの streamplot を使う例 (データがグリッドでないため)
            # 注: streamplotはグリッドデータを要求するため、元のデータ形式
            # (vector.dat) がグリッドでないと、こちらも実行が難しい。
            print("  (Streamplotは元データがグリッド形式でないためスキップします)")

except Exception as e:
    print(f"Tutorial 3 失敗: {e}")


# =======================================================
# Tutorial 4: PSD (2Dヒストグラム) (psd.dat)
# =======================================================
print("\nTutorial 4: PSD (2D Histogram)")
try:
    psd_data = file_read(os.path.join(DATA_DIR, 'psd.dat'))
    if psd_data is not None:
        data1 = psd_data[:, 0]
        data2 = psd_data[:, 1]
        
        psd, xax, yax = psd_calc(data1, data2, nbin_x=50, nbin_y=50)
        
        plt.figure(figsize=(10, 8))
        # psdは (nx, ny) で返る。imshowは (row, col) なので転置(.T)
        # extentで軸の値を設定
        plt.imshow(psd.T, origin='lower', aspect='auto', cmap='jet',
                   extent=[xax.min(), xax.max(), yax.min(), yax.max()])
        plt.colorbar(label='Counts')
        plt.xlabel('Data 1 (X-axis)')
        plt.ylabel('Data 2 (Y-axis)')
        plt.title('Tutorial 4: PSD (psd.dat)')
        plt.savefig('tutorial_4_psd.png')
        plt.close()
        print("-> tutorial_4_psd.png を保存しました。")
except Exception as e:
    print(f"Tutorial 4 失敗: {e}")

# =======================================================
# Tutorial 5: 1D FFT (fft.dat)
# =======================================================
print("\nTutorial 5: 1D FFT")
try:
    fft_data = file_read(os.path.join(DATA_DIR, 'fft.dat'))
    if fft_data is not None:
        time = fft_data[:, 0]
        data = fft_data[:, 1]
        
        # 順変換
        ans_shifted, freq_shifted = fftf(data, time, direction=-1)
        
        # パワースペクトル
        power = np.abs(ans_shifted)**2
        
        plt.figure(figsize=(10, 8))
        plt.plot(freq_shifted, power, color='red', linewidth=2)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.title('Tutorial 5: 1D FFT Power Spectrum (fft.dat)')
        plt.yscale('log') # IDL同様、対数プロット
        plt.savefig('tutorial_5_fft.png')
        plt.close()
        print("-> tutorial_5_fft.png を保存しました。")
except Exception as e:
    print(f"Tutorial 5 失敗: {e}")

# =======================================================
# Tutorial 6: 2D FFT (fft2d.dat)
# =======================================================
print("\nTutorial 6: 2D FFT")
try:
    # fft2d.dat は (nx, ny) のグリッドデータと仮定
    data2d_fft = file_read(os.path.join(DATA_DIR, 'fft2d.dat'))
    if data2d_fft is not None:
        nx, nt = data2d_fft.shape
        x = np.arange(nx)
        t = np.arange(nt)
        
        ans_shifted, wnum_shifted, freq_shifted = fftf2d(data2d_fft, x, t, direction=-1)
        
        # 2D パワースペクトル (Logスケール)
        power2d = np.log10(np.abs(ans_shifted)**2)
        
        plt.figure(figsize=(10, 8))
        # IDLの (col, row) -> Numpy (row, col) のため転置
        plt.imshow(power2d.T, origin='lower', aspect='auto', cmap='jet',
                   extent=[wnum_shifted.min(), wnum_shifted.max(), 
                           freq_shifted.min(), freq_shifted.max()])
        plt.colorbar(label='Log10(Power)')
        plt.xlabel('Wavenumber (k)')
        plt.ylabel('Frequency (f)')
        plt.title('Tutorial 6: 2D FFT Spectrum (fft2d.dat)')
        plt.savefig('tutorial_6_fft2d.png')
        plt.close()
        print("-> tutorial_6_fft2d.png を保存しました。")
except Exception as e:
    print(f"Tutorial 6 失敗: {e}")

print("\n--- 可視化処理が完了しました。---")

# poisson_bp.py は、この可視化チュートリアルでは使用されません。
# これはポアソン方程式を解くための計算モジュールです。