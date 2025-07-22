import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from kan import KANLinear
import torchvision.models as models


class ImgNet(nn.Module):
    def __init__(self, n_class):
        super(ImgNet, self).__init__()
        self.n_class = n_class
        img_model = models.resnext50_32x4d(pretrained=False)  # resnext50_32x4d  # resnext101_32x8d
        self.img_model = list(img_model.children())[:-1]
        self.img_model = nn.Sequential(*self.img_model)

        self.fc = nn.Linear(2048, 256)
        self.activation = nn.ReLU()
        self.last_fc = nn.Linear(256, self.n_class)

    def forward(self, img):
        img = self.img_model(img)
        img = img.view(img.size(0), -1)
        img = self.fc(img)
        img = self.activation(img)
        img_fc = self.last_fc(img)
        return img_fc


class MMFMixer_mission(nn.Module):
    def __init__(self, n_class):
        super(MMFMixer_mission, self).__init__()
        self.n_class = n_class
        out_dim = 128
        self.kan_768 = KANLinear(in_features=768, out_features=out_dim)
        self.kan_128 = KANLinear(in_features=128, out_features=out_dim)
        self.kan_23 = KANLinear(in_features=23, out_features=out_dim)
        self.kan_20 = KANLinear(in_features=20, out_features=out_dim)
        self.rs_model = ImgNet(out_dim)
        self.fc = KANLinear(in_features=out_dim * 5, out_features=out_dim)
        self.lastfc = KANLinear(in_features=128, out_features=self.n_class)
        self.activation = nn.ReLU()

        # 初始化提示向量
        self.prompt_vector_missing_doc = nn.Parameter(torch.randn(1, 768))
        self.prompt_vector_missing_graph = nn.Parameter(torch.randn(1, 128))
        self.prompt_vector_missing_sv = nn.Parameter(torch.randn(1, 20))

        # 将提示向量初始化为正态分布
        trunc_normal_(self.prompt_vector_missing_doc, std=.02)
        trunc_normal_(self.prompt_vector_missing_graph, std=.02)
        trunc_normal_(self.prompt_vector_missing_sv, std=.02)

    def forward(self, doc=None, graph=None, poi_l=None, rs=None, sv=None):
        poi_f1 = self.kan_23(poi_l)
        rs_f = self.rs_model(rs)

        # 将提示向量扩展到批量大小
        prompt_missing_doc = self.prompt_vector_missing_doc
        prompt_missing_graph = self.prompt_vector_missing_graph
        prompt_missing_sv = self.prompt_vector_missing_sv

        # 如果模态缺失，则使用对应的提示向量
        has_all_zero_row_doc = torch.all(doc == 0, dim=1)
        if has_all_zero_row_doc.any():
            doc[has_all_zero_row_doc] = prompt_missing_doc

        has_all_zero_row_graph = torch.all(graph == 0, dim=1)
        if has_all_zero_row_graph.any():
            graph[has_all_zero_row_graph] = prompt_missing_graph

        has_all_zero_row_sv = torch.all(sv == 0, dim=1)
        if has_all_zero_row_sv.any():
            sv[has_all_zero_row_sv] = prompt_missing_sv

        doc_f = self.kan_768(doc)
        graph_f = self.kan_128(graph)
        sv_f = self.kan_20(sv)
        x = torch.cat([doc_f, graph_f, rs_f, sv_f, poi_f1], dim=1)

        patch_mixer = self.fc(x)
        patch_mixer = self.activation(patch_mixer)
        patch_mixer = self.lastfc(patch_mixer)

        return patch_mixer
