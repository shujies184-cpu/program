"""
作者: 因吉
邮箱: inki.yinji@gmail.com
创建日期：2020 0903
近一次修改：2021 0714
说明：多示例学习的原型文件，用于获取数据集名称、包空间、包大小等
"""

import warnings
import numpy as np
import os as os
from Code.Function import load_file
warnings.filterwarnings("ignore")


class MIL:

    """
    多示例学习的原型类
    :param
        data_path：   数据集的存储路径
        save_home：   中间数据存储路径，例如：存储距离矩阵的目录
        bag_space：   格式与.mat文件一致
                      需要注意的是，当bag_space为None时，将读取给定目录下的文件
                      否则，将使用bag_space中的数据，但是依然需要传递文件名，以获取相对应的距离矩阵
    :attributes
        data_name：   数据集的名称
        bag_space：   包空间，详细格式请查看../Data/Benchmark/musk1+.mat
        ins_space：   实例空间
        bag_size：    记录每个包大小的向量，长度为N
        bag_lab：     包标签向量
        ins_lab：     实例标签
        bag_idx：     包索引向量（索引是什么？）
        ins_idx：     实例空间中 包所对应的实例的范围★★★
        ins_bag_idx： 实例空间中 实例对应的包的序号★
        zero_ratio：  数据集含零比率（实例特征矩阵中值为0的元素占所有特征元素的比例）
        N：           包空间的大小
        n：           实例数量
        d：           实例的维度
        C：           数据集的类别数
    """
    def __init__(self, data_path, save_home="../Data/Distance/", bag_space=None):
        self.data_path = data_path
        self.save_home = save_home
        self.bag_space = bag_space
        self.__init_mil()

    def __init_mil(self):
        """
        初始化函数
        """
        if self.bag_space is None:
            self.bag_space = load_file(self.data_path)
        self.N = len(self.bag_space)

        self.bag_size = np.zeros(self.N, dtype=int)#创建指定形状、指定数据类型的全 0 数组的核心函数（整数→一维数组;元组→多维数组）
        self.bag_lab = np.zeros_like(self.bag_size, dtype=int)#np.zeros_like()与np.zeros()相似，但是np.zeros_like()会根据现有数组的形状和类型创建零数组

        self.bag_idx = np.arange(self.N)    #np.arange 能按「起始值、终止值、步长」生成一个左闭右开的数值序列
        for i in range(self.N):
            self.bag_size[i] = len(self.bag_space[i][0])
            self.bag_lab[i] = self.bag_space[i][1]
        # 将所有包的标签调整到 [0, C - 1]的范围，C为数据集的类别数
        self.__bag_lab_map()

        self.n = sum(self.bag_size)
        self.d = len(self.bag_space[0, 0][0]) - 1
        self.C = len(list(set(self.bag_lab)))

        self.ins_space = np.zeros((self.n, self.d))
        self.ins_idx = np.zeros(self.N + 1, dtype=int)
        self.ins_lab = np.zeros(self.n)
        self.ins_bag_idx = np.zeros(self.n, dtype=int)
        for i in range(self.N):#★（需要再看看）这段代码是多示例学习（MIL）中「包级数据」向「实例级数据」转换的核心逻辑，目的是把分散在每个包中的实例特征、标签、所属关系，统一整合到全局的实例空间中
            self.ins_idx[i + 1] = self.bag_size[i] + self.ins_idx[i]
            self.ins_space[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bag_space[i, 0][:, :self.d]
            self.ins_lab[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bag_space[i, 0][:, -1]
            self.ins_bag_idx[self.ins_idx[i]: self.ins_idx[i + 1]] = np.ones(self.bag_size[i]) * i

        self.data_name = self.data_path.strip().split("/")[-1].split(".")[0]
        self.zero_ratio = len(self.ins_space[self.ins_space == 0]) / (self.n * self.d)
        self.__generate_save_home()

    def __generate_save_home(self):
        """
        Generate the save home.
        """
        if not os.path.exists(self.save_home):
            os.makedirs(self.save_home)

    def __bag_lab_map(self):#★★★这个函数的目标是消除标签格式差异，统一编码规则，而且多示例学习也不局限于是二分类问题；
        """
        Map the label of the bag to class \in [0, 1, 2, ...]
        """
        lab_list = list(set(self.bag_lab))
        lab_dict = {}
        for i, lab in enumerate(lab_list):
            lab_dict[lab] = i
        for i in range(self.N):
            self.bag_lab[i] = lab_dict[self.bag_lab[i]]

    def get_data_info(self):
        """
        Print the data set information.
        """
        temp_idx = 5 if self.N > 5 else self.N
        print("The {}'s information is:".format(self.data_name), "\n"
              "Number bags:", self.N, "\n"
              "Number classes:", self.C, "\n"
              "Bag size:", self.bag_size[:temp_idx], "...\n"
              "Bag label", self.bag_lab[:temp_idx], "...\n"
              "Maximum bag's size:", np.max(self.bag_size), "\n"
              "Minimum bag's size:", np.min(self.bag_size), "\n"
              "Zero ratio:", self.zero_ratio, "\n"
              "Number instances:", self.n, "\n"
              "Instance dimensions:", self.d, "\n"
              "Instance index:", self.ins_idx[: temp_idx], "...\n"
              "Instance label:", self.ins_lab[: temp_idx], "...\n"
              "Instance label corresponding bag'S index:", self.ins_bag_idx[:temp_idx], "...\n")

    def get_sub_ins_space(self, bag_idx):#按 “包索引” ，函数会提取这些包包含的所有实例
        """
        Given a bag idx array, and return a subset of instance space.
        """
        n = sum(self.bag_size[bag_idx])
        ret_ins_space = np.zeros((n, self.d))
        ret_ins_label = np.zeros(n)
        ret_ins_bag_idx = np.zeros(n, dtype=int)
        count = 0
        for i in bag_idx:
            bag_size = self.bag_size[i]
            ret_ins_space[count: count + bag_size] = self.bag_space[i, 0][:, :-1]
            ret_ins_label[count: count + bag_size] = self.bag_lab[i]
            ret_ins_bag_idx[count: count + bag_size] = i
            count += bag_size

        return ret_ins_space, ret_ins_label, ret_ins_bag_idx


if __name__ == '__main__':
    temp_file_name = r"..\Data\Benchmark\fox+.mat"
    mil = MIL(temp_file_name)
    mil.get_data_info()
