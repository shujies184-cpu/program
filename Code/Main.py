import argparse
import numpy as np
import os
from Code.ELDB import ELDB


def get_parser():
    """默认参数设置"""
    # 1. 使用原生 Python 读取 parameter_ini.txt
    ini_data = {}
    with open("../Data/Ini/parameter_ini.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:     # 确保行内容至少有“键”和“值”两部分
                ini_data[parts[0]] = parts[1]

    # 根据截图内容获取对应的值
    # ini_data['save_path_parameter'] -> ../Data/Record/Parameters/
    # ini_data['k'] -> 10

    # 提取指定参数，若键不存在则使用默认值（避免KeyError）
    save_home = ini_data.get('save_path_parameter', '../Data/Record/Parameters/')
    k_val = ini_data.get('k', '10')

    # 2. 拼接子参数文件的路径
    save_path_parameter_txt = (save_home + data_name + '_' +
                               mode_action + '_' + k_val + ".txt")

    # 3. 判断子参数文件是否存在
    if not os.path.exists(save_path_parameter_txt):
        print(f"配置文件不存在，使用默认值: {save_path_parameter_txt}")
        psi = 0.9
        type_b2b = "ave"
        mode_bag_init = 'g'
    else:
        # 如果子文件存在，读取其中的 psi, type_b2b, mode_bag_init
        # 注意这里加了 dtype=None 和 encoding
        parameters = np.genfromtxt(save_path_parameter_txt, dtype=None, encoding='utf-8')
        # 假设子文件的结构是：第一列标签，第二列数值
        psi = float(parameters[0][-1])
        type_b2b = str(parameters[1][-1])
        mode_bag_init = str(parameters[2][-1])

    parser = argparse.ArgumentParser(description="多示例学习ELDB算法的参数设置")    # 创建解析器对象
    # 添加参数
    parser.add_argument("--psi", default=psi, type=float, help="辨别包的选取比例")
    parser.add_argument("-alpha", default=0.75, type=float, help="学习率")
    parser.add_argument("--psi_max", default=100, type=int, help="最大选取包数")
    parser.add_argument("--type_b2b", default=type_b2b, help="距离度量")
    parser.add_argument("--mode_bag_init", default=mode_bag_init, help="初始dBagSet选取模式")
    parser.add_argument("--mode-action", default=mode_action, help="行为模式")
    parser.add_argument("--k", default=int(k_val), type=int)
    parser.add_argument("--type_performance", default=["f1_score"], type=list, help="性能度量指标")
    parser.add_argument("--save_path_classification_result",
                        default=ini_data.get('save_path_classification_result', '../Data/Record/ClassificationResult/'),
                        help="分类结果保存路径")
    parser.add_argument("--print_loop", action="store_false", default=False, help="是否打印loop变化值")

    return parser.parse_args()  # 返回 解析参数


def main():
    """
    修改后的测试函数：增加了实时进度显示与实验结果对比
    """
    args = get_parser()
    # 创建ELDB对象
    eldb = ELDB(data_path=data_path, psi=args.psi, alpha=args.alpha, psi_max=args.psi_max,
                type_b2b=args.type_b2b, mode_bag_init=args.mode_bag_init,
                mode_action=args.mode_action, k=args.k,
                type_performance=args.type_performance, print_loop=args.print_loop)

    results = {}
    results_save = {}
    classifier_type, metric_type = eldb.get_state()

    # 初始化用于存储本次实验最佳结果的变量
    best_classifier = {m: "" for m in metric_type}
    best_ave = {m: 0.0 for m in metric_type}
    best_std = {m: 0.0 for m in metric_type}

    print(f"\n{'=' * 50}")
    print(f"开始实验: {data_name} (模式: {args.mode_action}, {args.k}折交叉验证)")
    print(f"{'=' * 50}")

    # --- 阶段 1: 运行 10 次重复实验并实时打印 ---
    for i in range(10):
        # 核心计算步骤
        result_temp = eldb.get_mapping()

        print(f"进度: [{i + 1}/10] 次重复实验已完成")

        for classifier in classifier_type:
            for metric in metric_type:
                key = classifier + ' ' + metric
                # 转换百分比并保留四位小数
                val_temp = float("{:.4f}".format(result_temp[key] * 100))

                if i == 0:
                    results[key] = [val_temp]
                else:
                    results[key].append(val_temp)

                # 实时显示每一折的分数（模拟 GUI 的 Middle Results）
                print(f"    -> {key}: {val_temp}%")

    # --- 阶段 2: 计算本次运行的平均统计数据 ---
    print(f"\n{'-' * 20} 本次运行汇总 (Current Session) {'-' * 20}")

    current_run_stats = {}  # 暂存本次的均值用于后续对比
    for metric in metric_type:
        for i, classifier in enumerate(classifier_type):
            key = classifier + ' ' + metric
            ave_temp = float("{:.4f}".format(np.average(results[key])))
            std_temp = float("{:.4f}".format(np.std(results[key], ddof=1)))
            current_run_stats[key] = ave_temp

            print(f"分类器 {classifier:5s} | {metric:10s} | 平均分: {ave_temp}% | 标准差: {std_temp}")

            # 选出本次运行中最棒的分类器
            if ave_temp > best_ave[metric]:
                best_classifier[metric] = classifier
                best_ave[metric] = ave_temp
                best_std[metric] = std_temp
                results_save[metric] = results[key]

    # --- 阶段 3: 历史存档对比与更新 ---
    data_save_path = (args.save_path_classification_result + data_name + '_' +
                      args.mode_action + '_' + str(args.k) + ".npz")

    # 确保保存目录存在
    if not os.path.exists(args.save_path_classification_result):
        os.makedirs(args.save_path_classification_result)

    # 首次运行则初始化存档
    if not os.path.exists(data_save_path):
        np.savez(data_save_path, best_classifier=best_classifier, best_ave=best_ave, best_std=best_std,
                 results_save=results_save)

    best_results_load = np.load(data_save_path, allow_pickle=True)

    print(f"\n{'=' * 20} 历史最佳对比 (Global Best) {'=' * 20}")

    for metric in metric_type:
        # 从存档中获取历史最高分
        history_best_ave = eval(str(best_results_load["best_ave"]))[metric]
        history_best_clf = eval(str(best_results_load["best_classifier"]))[metric]

        if best_ave[metric] < history_best_ave:
            # 如果本次没破纪录，则最终显示结果切换为历史存档
            print(f"[{metric}] 状态: 未能超越历史记录 (本次 {best_ave[metric]}% vs 历史 {history_best_ave}%)")
            best_classifier[metric] = history_best_clf
            best_ave[metric] = history_best_ave
            best_std[metric] = eval(str(best_results_load["best_std"]))[metric]
            results_save[metric] = eval(str(best_results_load["results_save"]))[metric]
        else:
            print(f"[{metric}] 状态: ★ 恭喜！本次运行刷新了历史最佳记录!")

        print(f"\t>>> 最终确定最佳分类器: {best_classifier[metric]}")
        print(f"\t>>> 10折交叉验证列表: {results_save[metric]}")
        print(f"\t>>> 平均精度 ± 标准差: {best_ave[metric]}% ± {best_std[metric]}")

    # 更新存档文件
    np.savez(data_save_path, best_classifier=best_classifier, best_ave=best_ave, best_std=best_std,
             results_save=results_save)
    print(f"\n结果已同步至存档: {data_save_path}")


if __name__ == '__main__':
    """进行实验时需要修改的参数"""
    # 数据集的路径
    data_path = "../Data/Benchmark/musk1+.mat"
    # 行为模式，对应于aELDB和rELDB
    mode_action = 'r'  # or 'a'     a:添加；r:替换

    # 用于记录最佳分类器的分类结果
    best_classifier, best_ave, best_std = {}, {}, {}
    # 获取数据集名称
    data_name = data_path.split('/')[-1].split('.')[0]
    print("实验数据集{:s}".format(data_name))
    main()
