import os
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import pandas as pd
import shapefile as shp
from tqdm import tqdm
import random

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        try:
            i_image = Image.open(image_path)
        except:
            continue
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == "__main__":
    data = {'idx': [], 'points': [], 'captioning': []}
    datasource = shp.Reader(r'H:\功能区论文\街景点\Wuhan_points_20m.shp')

    points = {}
    for feature in datasource.iterShapeRecords():
        idx = feature.record['idx']
        if idx not in points:
            points[idx] = []
        points[idx].append(feature.record['Doc_name'])
    for i in points:
        data['idx'].append(i)
        if len(points[i]) > 3:
            data['points'].append(random.sample(points[i], 3))
        else:
            data['points'].append(points[i])

    path = r'J:\街景数据\武汉市\武汉市街景图像'

    for i in tqdm(range(len(data['idx']))):
        image_paths = []
        for point in data['points'][i]:
            years = []
            for file in os.listdir(os.path.join(path, point)):
                year = int(file.split("_")[0][0:4])
                if year not in years:
                    years.append(year)
            year_need = max(years)
            for file in os.listdir(os.path.join(path, point)):
                if int(file.split("_")[0][0:4]) == year_need:
                    image_path = os.path.join(path, point, file)
                    image_paths.append(image_path)
        data['captioning'].append(predict_step(image_paths))

    df = pd.DataFrame(data)
    df.to_csv(r'H:\功能区论文\特征\image_captioning_new\Wuhan_all.csv', index=True)
