import torch
import numpy as np

def load_embeddings(file_name):
    model = torch.load(file_name, map_location=torch.device('cpu'))  # Load the PyTorch model
    word_vectors = model['vectors']
    vocabulary = model['vocabulary']
    return word_vectors, vocabulary

def outputTxt(embeddings_file):
    embeddings_file = embeddings_file  # PyTorch model file address
    word_vectors, vocabulary = load_embeddings(embeddings_file)
    matrix_txt_file = embeddings_file + ".txt"  # Matrix txt file address

    with open(matrix_txt_file, 'w') as out:
        for word, vector in zip(vocabulary, word_vectors):
            out.write(str(word) + "\t")
            out.write(" ".join(str(e) for e in vector.tolist()))
            out.write("\n")
