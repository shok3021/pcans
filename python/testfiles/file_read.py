import numpy as np
import glob
import gzip
import warnings

def file_read(filename_pattern, dtype=float, delimiter=None, silent=False, compress_flag=False):
    """
    IDLのfile_read.proのPython (Numpy) 版。
    
    Args:
        filename_pattern (str): globパターン (例: 'data_*.dat')
        compress_flag (bool): IDLのcompressキーワード。
                              Trueの場合、ファイル名に関わらずgzipとして開こうと試みる。
    """
    flist = sorted(glob.glob(filename_pattern))
    
    if not flist:
        print("No such file")
        return 0

    all_data = []
    for l, f in enumerate(flist):
        if not silent:
            print(f"{l}.  {f}  Reading......")
        
        try:
            # compressキーワードがTrueならgzip.open、
            # Falseならファイル名が .gz で終わる場合のみloadtxtが自動解凍
            if compress_flag:
                with gzip.open(f, 'rt') as file_obj:
                    data = np.loadtxt(file_obj, dtype=dtype, delimiter=delimiter)
            else:
                data = np.loadtxt(f, dtype=dtype, delimiter=delimiter)

            if not silent and l == 0:
                if data.ndim == 0:
                    print("column: 1, line: 1")
                elif data.ndim == 1:
                    # 1行のデータか、1列のデータか判別が難しいが、
                    # 1行のデータとしておく
                    print(f"column: {data.shape[0]}, line: 1") 
                else:
                    print(f"column: {data.shape[1]}, line: {data.shape[0]}")
                        
            all_data.append(data)
        
        except Exception as e:
            print(f"Error reading {f}: {e}")
            if l == 0: return 0

    if not all_data:
        return 0
    
    # 複数のファイルを読み込んだ場合、(count, line, col) の3D配列にする
    try:
        result = np.array(all_data)
        # ファイルが1つなら (line, col)、複数なら (count, line, col)
        return np.squeeze(result)
    except ValueError as e:
        # 配列の形状が異なる場合
        warnings.warn(f"Could not stack arrays into a single NumPy array: {e}. Returning list.")
        return all_data