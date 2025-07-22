import os
import joblib
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
from tqdm import tqdm
import random
from graph_embeddings import convert_to_list


# Step 1: Feature extraction using SIFT
def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()
    all_idx = []
    descriptors_list = []
    for path in image_paths:
        # 读取.tif格式的图像并转换为灰度图像
        with Image.open(path) as img:
            gray_img = img.convert("L")
            image = np.array(gray_img)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None and descriptors.shape[1] == 128:
            all_idx.append(path.split("\\")[-1].split(".")[0])
            descriptors_list.append(descriptors)
    return descriptors_list, all_idx


# Step 2: Generate visual vocabulary using K-means
def train_kmeans(descriptors_list, k, model_path):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=k, n_init='auto').fit(all_descriptors)
    # 保存模型到文件
    joblib.dump(kmeans, model_path)
    # visual_vocabulary = kmeans.cluster_centers_
    # return visual_vocabulary, kmeans


# Step 3: Generate visual word frequency vector for each image
def generate_bow_vectors(descriptors_list, kmeans):
    bow_vectors = []
    for descriptors in descriptors_list:
        histogram = np.zeros(len(kmeans.cluster_centers_))
        cluster_assignments = kmeans.predict(descriptors)
        for idx in cluster_assignments:
            histogram[idx] += 1
        bow_vectors.append(histogram)
    return np.array(bow_vectors)


def train(image_paths, k, model_path):
    descriptors_list, all_idx = extract_sift_features(image_paths)
    train_kmeans(descriptors_list, k, model_path)


def predict(image_paths, model_path, out_path):
    descriptors_list, all_idx = extract_sift_features(image_paths)
    # 从文件加载模型
    kmeans = joblib.load(model_path)
    bow_vectors = generate_bow_vectors(descriptors_list, kmeans)

    df = pd.DataFrame(bow_vectors)
    df['idx'] = all_idx
    cols = ['idx'] + df.columns[: -1].tolist()
    df = df[cols]
    df.to_csv(out_path, index=True)

    # print("Visual Vocabulary:\n", visual_vocabulary)
    # print("BoW Vectors:\n", bow_vectors)


def predict_street_view():
    # 从文件加载模型
    kmeans = joblib.load('./street_view.pkl')

    dic = {'长沙市': 'Changsha', '武汉市': 'Wuhan', '南昌市': 'Nanchang'}
    parcels_pictures = {}
    for city in ['长沙市', '南昌市', '武汉市']:
        path = f'I:\\街景数据\\{city}\\{city}街景图像'
        parcels_data = pd.read_csv(f'H:\\功能区论文\\特征\\image_captioning\\{dic[city]}_all.csv')
        parcels_data['points'] = parcels_data['points'].apply(convert_to_list)
        parcels_fid = parcels_data['idx'].values.tolist()
        parcels_points = {}
        for idx in parcels_fid:
            parcels_points[idx] = parcels_data[parcels_data['idx'] == idx]['points'].values[0]

        for idx in parcels_points:
            parcels_pictures[idx] = []
            for point in parcels_points[idx]:
                if point == '27920_112.790049886,28.346729326':
                    continue
                for image in os.listdir(os.path.join(path, point)):
                    parcels_pictures[idx].append(os.path.join(path, point, image))

    parcels_bovw = {}
    for idx in tqdm(parcels_pictures):
        image_paths = parcels_pictures[idx]
        descriptors_list, all_idx = extract_sift_features(image_paths)
        bow_vectors = generate_bow_vectors(descriptors_list, kmeans)
        df = pd.DataFrame(bow_vectors)
        sum_df = df.sum(axis=0).to_frame().T / len(image_paths)
        parcels_bovw[idx] = sum_df

    # 初始化一个空列表，用于存储所有带有 idx 列的 DataFrame
    df_list = []

    # 遍历字典，将每个 DataFrame 添加一列 idx
    for idx, df in parcels_bovw.items():
        df['idx'] = idx
        df_list.append(df)

    # 将所有 DataFrame 合并为一个 DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # 将 idx 列移动到第一列
    cols = combined_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    combined_df = combined_df[cols]

    # 将合并后的 DataFrame 写入 CSV 文件
    combined_df.to_csv(r'H:\功能区论文\特征\BOVW_street_view\output.csv', index=True)


if __name__ == "__main__":
    image_paths = []
    for file in os.listdir(r'G:\Bag-of-Visual-Words-master\Code Files\data\images'):
        image_paths.append(os.path.join(r'G:\Bag-of-Visual-Words-master\Code Files\data\images', file))
    predict(image_paths, './remote_sensing.pkl', r'H:\功能区论文\特征_new\BOVW_remote_sensing\remote_sensing.csv')