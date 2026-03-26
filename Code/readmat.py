import scipy.io as sio
import numpy as np

# 1. 读取.mat文件
mat_file_path = "..\Data\Benchmark\elephant+.mat"  # 替换为实际文件路径
mat_data = sio.loadmat(mat_file_path)

# 2. 查看.mat文件中的变量名（返回的是字典，键为变量名）
print(".mat文件中的变量名：", mat_data.keys())

# 3. 提取指定变量的数据（例如变量名为'data'）
# 注意：scipy读取的结果是numpy数组格式，可直接用于数值计算
if 'data' in mat_data:
    data = mat_data['data']
    print("提取的数据形状：", data.shape)  # 查看数组维度
    print("数据前1行：\n", data[:1])      # 打印部分数据验证
else:
    print("未找到指定变量，请检查变量名！")