import os
import torch
from torch.utils import data
from torchvision import transforms
from dataset import MyDataset
import argparse
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kappa_from_confusion_matrix(conf_matrix):
    # 计算总样本数
    total_samples = np.sum(conf_matrix)

    # 计算观察到的准确率 Po
    Po = np.trace(conf_matrix) / total_samples

    # 计算随机预测下的准确率 Pe
    Pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total_samples ** 2)

    # 计算 Kappa
    kappa = (Po - Pe) / (1 - Pe)

    return kappa


def val_MMF(args):
    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = MyDataset(path=r'./data/features', shp_path=r'./data/new/test.shp',
                            images_path=r'G:\Bag-of-Visual-Words-master\Code Files\data\images', transform=my_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = torch.load("./model_20250209/MMF-6.pth")
    model = model.to(device)
    model.eval()

    classes = ("居住区", "商业区", "工业区", "公共服务区")
    class_correct = list(0. for i in range(args.num_class))
    class_total = list(0. for i in range(args.num_class))
    correct_prediction = 0.
    total = 0

    # 初始化变量来存储所有的实际标签和预测标签
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for doc_e, graph_e, poi_l, poi_g, rs_f, sv_f, labels in val_loader:
            labels = labels.to(device)
            doc_e = doc_e.to(device)
            graph_e = graph_e.to(device)
            poi_l = poi_l.to(device)
            # poi_g = poi_g.to(device)
            rs_f = rs_f.to(device)
            sv_f = sv_f.to(device)

            outputs = model(doc_e, graph_e, poi_l, rs_f, sv_f)

            _, predicted = torch.max(outputs, 1)
            # 保存当前批次的真实标签和预测标签
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            c = (predicted == labels).squeeze()
            for label_idx in range(len(labels)):
                label = labels[label_idx]
                class_correct[label] += c[label_idx].item()
                class_total[label] += 1
            total += labels.size(0)
            correct_prediction += (predicted == labels).sum().item()

    for i in range(args.num_class):
        print('Model - Accuracy of %5s : %2d %%: Correct Num: %d in Total Num: %d' % (
            classes[i], 100 * class_correct[i] / class_total[i], class_correct[i], class_total[i]))
    acc = correct_prediction / total
    print("Total Acc Model: %.4f" % acc)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("混淆矩阵：")
    print(conf_matrix)
    return conf_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='val hyper-parameter')
    parser.add_argument("--num_class", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    conf_matrix = val_MMF(args)
    # 1. 计算每一行的总和 (即每个真实类别的样本总数)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)

    # 2. 计算百分比混淆矩阵（每个元素除以该行的总和，并乘以 100）
    percent_conf_matrix = conf_matrix / row_sums
    classes = ("R. Z", "C. Z", "I. Z", "P.S. Z")
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(percent_conf_matrix, annot=True, fmt=".3f", cmap="viridis", xticklabels=classes,
                     yticklabels=classes,
                     annot_kws={"size": 34}, square=True)
    # 获取 color bar 对象
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=18)  # 设置 color bar 刻度标签的字体大小
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 调整 xticks 和 yticks 字体大小
    plt.xticks(fontsize=26)  # 设置 x 轴类标签字体大小
    plt.yticks(fontsize=26)  # 设置 y 轴类标签字体大小

    # 调整 xlabel 和 ylabel 字体大小
    plt.xlabel('Predicted Label', fontsize=26)  # 设置 x 轴标签字体大小
    plt.ylabel('True Label', fontsize=26)  # 设置 y 轴标签字体大小
    plt.show()
    kappa = kappa_from_confusion_matrix(conf_matrix)
    print(f"Kappa 系数: {kappa}")
