from sentence_transformers import SentenceTransformer
import pandas as pd
import shapefile as shp
from graph_embeddings import convert_to_list


def bert_embeddings(city, flag):
    parcels_sv = pd.read_csv(f'H:\\功能区论文\\特征\\image_captioning\\{city}_{flag}_label.csv')
    parcels_fid = parcels_sv['idx'].values.tolist()
    parcels_sv['captioning'] = parcels_sv['captioning'].apply(convert_to_list)
    scene_list = parcels_sv['captioning'].values.tolist()

    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    embeddings = model.encode(scene_list)
    print(embeddings.shape)

    df = pd.DataFrame(embeddings)
    df['idx'] = parcels_fid
    cols = ['idx'] + df.columns[: -1].tolist()
    df = df[cols]
    df.to_csv(f'H:\\功能区论文\\特征\\doc_embeddings\\{city}_{flag}_label.csv', index=True)


def select_label_csv(city, flag):
    points_sv = pd.read_csv(f'H:\\功能区论文\\特征\\image_captioning\\{city}_all.csv')
    data_label = []
    datasource = shp.Reader(f'H:\\功能区论文\\街景点\\{city}_points_20m.shp')

    for feature in datasource.iterShapeRecords():
        idx = feature.record['idx']
        type_id = feature.record['type_id']
        if idx not in data_label and type_id > 0:
            data_label.append(idx)

    output_file = f'H:\\功能区论文\\特征\\image_captioning\\{city}_{flag}_label.csv'
    if flag == 'with':
        filtered_df = points_sv[points_sv['idx'].isin(data_label)]
    elif flag == 'without':
        filtered_df = points_sv[~points_sv['idx'].isin(data_label)]
    else:
        filtered_df = []
    filtered_df.to_csv(output_file, index=True)


def bert_embeddings_new(city):
    parcels_sv = pd.read_csv(f'H:\\功能区论文\\特征_new\\image_captioning\\{city}_all.csv')
    parcels_fid = parcels_sv['idx'].values.tolist()
    print(len(parcels_fid))
    parcels_sv['captioning'] = parcels_sv['captioning'].apply(convert_to_list)
    scene_list = parcels_sv['captioning'].values.tolist()
    print(len(scene_list))

    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    embeddings = model.encode(scene_list)
    print(embeddings.shape)

    df = pd.DataFrame(embeddings)
    df['idx'] = parcels_fid
    cols = ['idx'] + df.columns[: -1].tolist()
    df = df[cols]
    df.to_csv(f'H:\\功能区论文\\特征_new\\doc_embeddings\\{city}.csv', index=True)


if __name__ == "__main__":
    for city in ['Changsha']:
        bert_embeddings_new(city)