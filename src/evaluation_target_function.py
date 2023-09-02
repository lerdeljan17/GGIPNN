import itertools
import random
import numpy as np
import torch

msigdb_file = "msigdb.v6.1.symbols.gmt"  # Update with your msigdb file address
pathwayList = []
with open(msigdb_file, 'r') as readFile:
    for line in readFile:
        tmpList = line.split("\t")
        n = len(tmpList)
        if n > 52:
            continue
        pathwayList.append(line)
readFile.close()

def load_word_vectors(file_name):
    word_vectors = {}
    with open(file_name, 'r') as file:
        for line in file:
            if len(line.split()) == 2:  # Skip the header line
                continue
            parts = line.strip().split()
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            word_vectors[word] = torch.tensor(vector)
    return word_vectors

def similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm1 = torch.norm(vector1)
    norm2 = torch.norm(vector2)
    return dot_product / (norm1 * norm2)

def targetFunc(emb_w2v_file):
    geneEmbedDict = load_word_vectors(emb_w2v_file)

    paths_array = []  # Numerator in target function

    for pathway in pathwayList:
        geneList = []
        path_arr = []
        tmpList = pathway.split("\t")
        n = len(tmpList)
        for i in range(2, n):
            gene = tmpList[i]
            if gene in geneEmbedDict:
                geneList.append(gene)
        genePairs = list(itertools.combinations(geneList, 2))
        for pair in genePairs:
            sim = similarity(geneEmbedDict[pair[0]], geneEmbedDict[pair[1]])
            path_arr.append(sim)
        paths_array.append(sum(path_arr) / len(path_arr))
        tmpList.clear()
        geneList.clear()

    randArray = []  # Denominator in target function
    random.seed(35)
    random.shuffle(list(geneEmbedDict.keys()))
    genePairs = list(itertools.combinations(list(geneEmbedDict.keys())[:1000], 2))
    for pair in genePairs:
        sim = similarity(geneEmbedDict[pair[0]], geneEmbedDict[pair[1]])
        randArray.append(sim)
    genePairs.clear()
    print("------------")
    print(emb_w2v_file)
    path_mean = np.mean(paths_array)
    rand_mean = np.mean(randArray)
    print(path_mean, end="")
    print("\t", rand_mean)
    print(path_mean / rand_mean)
    print("------------")
    return path_mean / rand_mean

emb_w2v_file = "../pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
targetFunc(emb_w2v_file)
