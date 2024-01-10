import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from karateclub import DeepWalk
from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

df_1_sam_1 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_1@sample_1.csv',header =0)
df_2_sam_1 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_2@sample_1.csv', header=0)
df_1_sam_9 = pd.read_csv('C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_fixed_sentence_samples/User_1@sample_9.csv', header=0)



def calculate_averages(df):

    averages = df.groupby('Keys')['F3'].mean().reset_index()
    return averages

def average_f3_timing(df, row_id):
    total_timing = df['F3'].sum()
    avg_time = round(((df['F3'][row_id])/total_timing),3)

    return avg_time


def creating_graph(df):
    G = nx.MultiDiGraph()
    weights = []
    for i in range(len(df)):
            key_pair = df['Keys'][i]
            first = key_pair[:key_pair.find(',')]
            second = key_pair[key_pair.find(',')+1:]
            time = average_f3_timing(df,i) * 100
            G.add_edge(first, second,weight = time)
            weights.append(time)
    return G, weights

def getting_adj_mtx(G):
    node_labels = list(G.nodes())
    adj_matrix = nx.to_numpy_matrix(G, nodelist=node_labels)

    adj_df = pd.DataFrame(adj_matrix, index=node_labels, columns=node_labels)

    return adj_df

def visualise_graph(G, weights, i):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title(f'Fixed_test_graph_user_'+ str(i))
    #plt.hist([v for k,v in nx.degree(G)])
    pos = nx.spring_layout(G)
    nx.draw(G,pos,node_size=80,font_size=5, with_labels = True, width= weights)
    plt.show()


#df = calculate_averages(df_1_sam_1)
#Graph1, weights1 = creating_graph(df)
#visualise_graph(Graph1, weights1,1)

#df = calculate_averages(df_2_sam_1)
#Graph2, weights2 = creating_graph(df)
#visualise_graph(Graph2, weights2,2)

def creating_graph_4_embed(df):
    G = nx.MultiDiGraph()
    graphs = []
    key_to_int_map = {}  
    next_index = 0
    for i in range(len(df)):
        key_pair = df['Keys'][i]
        first = key_pair[:key_pair.find(',')]
        second = key_pair[key_pair.find(',')+1:]

        if first not in key_to_int_map:
            key_to_int_map[first] = next_index
            next_index += 1
        if second not in key_to_int_map:
            key_to_int_map[second] = next_index
            next_index += 1
            
        first_int = key_to_int_map[first]
        second_int = key_to_int_map[second]

        time = average_f3_timing(df,i) * 100
        G.add_edge(first_int, second_int,weight = time)
    
    graphs.append(G)

    return graphs

def embed_graphs(graphs, method, dim = 5):
    from karateclub.graph_embedding import Graph2Vec, FGSD, GL2Vec

    if method == "gl2vec":
        model = GL2Vec(dimensions = dim)
        model.fit(graphs)
        embed = model.get_embedding()

    elif method == "fgsd":
        model = FGSD(hist_bins = dim)
        model.fit(graphs)
        embed = model.get_embedding()

    elif method == 'graph2vec':
        model = Graph2Vec(dimensions = dim)
        model.fit(graphs)
        embed = model.get_embedding()

#     elif method =='features':
#         embed = fn.graph_features(graphs, fast)
        # feat_norm = StandardScaler().fit_transform(feat)
        # pca_feat = PCA(n_components=dim)
        # embed = pca_feat.fit_transform(feat_norm)

    return embed


df = calculate_averages(df_1_sam_1)
Graph1 = creating_graph_4_embed(df)
print(embed_graphs(Graph1, 'graph2vec'))

df = calculate_averages(df_2_sam_1)
Graph2 = creating_graph_4_embed(df)
print(embed_graphs(Graph2, 'graph2vec'))

def creating_graph_4_deepwalk(df):
    G = nx.MultiDiGraph()
    graphs = []
    key_to_int_map = {}  
    next_index = 0
    for i in range(len(df)):
        key_pair = df['Keys'][i]
        first = key_pair[:key_pair.find(',')]
        second = key_pair[key_pair.find(',')+1:]

        if first not in key_to_int_map:
            key_to_int_map[first] = next_index
            next_index += 1
        if second not in key_to_int_map:
            key_to_int_map[second] = next_index
            next_index += 1
            
        first_int = key_to_int_map[first]
        second_int = key_to_int_map[second]

        time = average_f3_timing(df,i) * 100
        G.add_edge(first_int, second_int,weight = time)

    return G

def deep_walk(G):
    model = DeepWalk(walk_length=500, dimensions=2) 
    model.fit(G) 
    embedding = model.get_embedding()

    return embedding

df = calculate_averages(df_1_sam_1)
Graph1 = creating_graph_4_deepwalk(df)
print(deep_walk(Graph1))
#print(mapping1)

df = calculate_averages(df_2_sam_1)
Graph2 = creating_graph_4_deepwalk(df)
print(deep_walk(Graph2))
#print(mapping2)




#print(deep_walk_df(Graph1,1,1))


#fig,ax1= plt.subplots(figsize = (10,7))


#ax1.set_facecolor('gainsboro')
#ax1.scatter(deep_walk(Graph1)[:,0], deep_walk(Graph1)[:,1], color='red', s=50)    
#ax1.scatter(deep_walk(Graph2)[:,0], deep_walk(Graph2)[:,1], color='blue', s=50)  
#plt.show()
