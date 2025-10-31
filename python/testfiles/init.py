import sys
import os
sys.path.append('./')
pcans_dir = os.getenv('PCANS_DIR')
if pcans_dir:
    # Python用のスクリプトディレクトリを指定
    sys.path.append(os.path.join(pcans_dir, 'python_scripts'))

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'Times New Roman', # 'Times' は環境依存
    'font.size': 24,
    'axes.linewidth': 2,       # xthick, ythick
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.direction': 'in',   # xtickdir: 1
    'ytick.direction': 'in',   # ytickdir: 1
    # 'dimension' は描画時に figsize=[10, 10] (インチ指定) などで指定
})

import os
# 例: 利用可能なCPUコアの半分に設定
# num_threads = max(1, os.cpu_count() // 2)
# os.environ['OMP_NUM_THREADS'] = str(num_threads)

# または、threadpoolctl を使用
from threadpoolctl import threadpool_limits
num_threads = max(1, os.cpu_count() // 2)
# with threadpool_limits(limits=num_threads, user_api='all'):
#     # この中でNumpy/Scipyの計算を実行
#     pass