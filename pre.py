import os
import torch
from torch.utils import data
from torchvision import transforms
from dataset import MyDataset, preDataset
import argparse
import shapefile as shp
from osgeo import osr
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
type_dic = {1: "居住区", 2: "商业区", 3: "工业区", 4: "公共服务区"}


def pre(args):
    my_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pre_dataset = preDataset(path=r'./data/features', shp_path=r'./data/test/test_30.shp',
                             images_path=r'G:\Bag-of-Visual-Words-master\Code Files\data\images',
                             transform=my_transform)

    pre_loader = torch.utils.data.DataLoader(pre_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = torch.load("./model_new_0919/MMF-4.pth")
    model = model.to(device)
    model.eval()
    predictions = []
    ids = []
    with torch.no_grad():
        for doc_e, graph_e, poi_c, rs_f, sv_f, keys in tqdm(pre_loader):
            doc_e = doc_e.to(device)
            graph_e = graph_e.to(device)
            poi_c = poi_c.to(device)
            rs_f = rs_f.to(device)
            sv_f = sv_f.to(device)

            outputs = model(doc_e, graph_e, poi_c, rs_f, sv_f)

            _, predicted = torch.max(outputs, 1)
            # 保存当前批次的真实标签和预测标签
            ids.extend(keys.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
    result_dict = dict(zip(ids, predictions))
    return result_dict


def create_shp(pre_dataset, shp_path, dic):
    output_shp = shp.Writer(shp_path)
    read_shp = shp.Reader(pre_dataset)
    for field in read_shp.fields[1:]:
        output_shp.field(*field)

    # 找到 "type" 字段的索引
    type_field_index = [f[0] for f in read_shp.fields].index("type") - 1
    typeId_field_index = [f[0] for f in read_shp.fields].index("type_id") - 1
    for shapeRec in read_shp.iterShapeRecords():
        modified_record = list(shapeRec.record)
        modified_record[type_field_index] = type_dic[dic[int(shapeRec.record['idx'])] + 1]
        modified_record[typeId_field_index] = dic[int(shapeRec.record['idx'])] + 1
        # 写入修改后的记录和形状
        output_shp.record(*modified_record)
        output_shp.shape(shapeRec.shape)

    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    wkt = proj.ExportToWkt()
    coding = "UTF-8"

    f = open(shp_path.replace(".shp", ".prj"), 'w')
    g = open(shp_path.replace(".shp", ".cpg"), 'w')
    g.write(coding)
    f.write(wkt)
    g.close()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre hyper-parameter')
    parser.add_argument("--num_class", default=4, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()
    dic = pre(args)
    read_shp = shp.Reader(r'./data/test/test_30.shp')
    dic_true = {}
    for shapeRecord in read_shp.iterRecords():
        if shapeRecord['type_id'] in [4, 5, 6, 7]:
            type_id = 3
        else:
            type_id = shapeRecord['type_id'] - 1
        dic_true[shapeRecord['idx']] = type_id
    dic_dif = {}
    for idx in dic:
        if dic[idx] != dic_true[idx]:
            dic_dif[idx] = dic[idx]
    print(dic_dif)
    create_shp(r'./data/pre/pre.shp', r'./data/pre/pre_label_new.shp', dic)