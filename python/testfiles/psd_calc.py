import numpy as np

def psd_calc(data1, data2, max_x=None, min_x=None, max_y=None, min_y=None, 
             nbin_x=20, nbin_y=20):
    """
    IDLのpsd_calcのPython (Numpy) 版
    """
    
    if max_x is None: max_x = np.max(data1)
    if min_x is None: min_x = np.min(data1)
    if max_y is None: max_y = np.max(data2)
    if min_y is None: min_y = np.min(data2)
    
    bins = [nbin_x, nbin_y]
    ranges = [[min_x, max_x], [min_y, max_y]]
    
    # psd: (nbin_x, nbin_y) の配列
    # xedges, yedges: (nbin+1) のビンの境界
    psd, xedges, yedges = np.histogram2d(data1, data2, bins=bins, range=ranges)
    
    # IDL版はビンの下限値を返しているので、numpyのedgesの[:-1] (最後の境界を除く) を返す
    xax = xedges[:-1]
    yax = yedges[:-1]
    
    # IDL (hist_2d) は (X, Y) の順
    # Numpy (histogram2d) も (X, Y) の順で返すが、
    # imshowなどで表示する際は (Y, X) の順 (psd.T) になる点に注意
    
    return psd, xax, yax