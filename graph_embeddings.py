import numpy as np
import pandas as pd
import spacy
from itertools import combinations
import networkx as nx
import ast
from Module import Graph2Vec

nlp = spacy.load('en_core_web_sm')


def get_entity(emp):
    emp = 'a truck driving down a city street near tall buildings'
    entity_list = []
    for tok in nlp(emp):
        if tok.pos_ in ['NOUN', 'VERB']:
            if tok.text == 'city':
                continue
            entity_list.append(tok.text)
    c = list(combinations(entity_list, 2))
    return c


def graph_embeddings(city, flag):
    parcels_sv = pd.read_csv(f'H:\\功能区论文\\特征\\image_captioning\\{city}_{flag}_label.csv')
    parcels_fid = parcels_sv['idx'].values.tolist()
    osm_sv_scene = {}
    parcels_sv['captioning'] = parcels_sv['captioning'].apply(convert_to_list)

    for idx in parcels_fid:
        osm_sv_scene[idx] = parcels_sv[parcels_sv['idx'] == idx]['captioning'].values[0]

    graph_embd = {}
    for i in parcels_fid:
        G = nx.Graph()
        nodes = []
        for j in osm_sv_scene[i]:
            tmp = get_entity(j)
            for l in tmp:
                nodes.append(l[0])
                nodes.append(l[1])
                G.add_edge(l[0], l[1])
        nodes = list(set(nodes))
        G.add_nodes_from(nodes)
        convertedgraph = nx.convert_node_labels_to_integers(G)
        embedding_model = Graph2Vec()
        embedding_model.fit([convertedgraph])
        embeddingsframe = pd.DataFrame(embedding_model.get_embedding())
        graph_embd[i] = embeddingsframe.values[0]
    df = pd.DataFrame(graph_embd).T
    df.to_csv(f'H:\\功能区论文\\特征\\graph_embeddings\\{city}_{flag}_label.csv', index=True)


def convert_to_list(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return x


def graph_embeddings_new(city):
    parcels_sv = pd.read_csv(f'H:\\功能区论文\\特征_new\\image_captioning\\{city}_all.csv')
    parcels_fid = parcels_sv['idx'].values.tolist()
    osm_sv_scene = {}
    parcels_sv['captioning'] = parcels_sv['captioning'].apply(convert_to_list)

    for idx in parcels_fid:
        osm_sv_scene[idx] = parcels_sv[parcels_sv['idx'] == idx]['captioning'].values[0]

    graph_embd = {}
    for i in parcels_fid:
        G = nx.Graph()
        nodes = []
        for j in osm_sv_scene[i]:
            tmp = get_entity(j)
            for l in tmp:
                nodes.append(l[0])
                nodes.append(l[1])
                G.add_edge(l[0], l[1])
        nodes = list(set(nodes))
        G.add_nodes_from(nodes)
        convertedgraph = nx.convert_node_labels_to_integers(G)
        embedding_model = Graph2Vec()
        embedding_model.fit([convertedgraph])
        embeddingsframe = pd.DataFrame(embedding_model.get_embedding())
        graph_embd[i] = embeddingsframe.values[0]
    df = pd.DataFrame(graph_embd).T
    df.to_csv(f'H:\\功能区论文\\特征_new\\graph_embeddings\\{city}_all.csv', index=True)


if __name__ == "__main__":
    print(get_entity(''))
