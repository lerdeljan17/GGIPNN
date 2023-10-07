import torch.nn as nn
import torch
import GGIPNN_util as NN_util

class GGIPNN(nn.Module):
    """
    A neural network for text classification.
    Uses an embedding layer, followed by a hidden and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, hidden_dimension=100, embedTrain = False, l2_lambda =0, 
        use_pre_trained_gene2vec = False, all_text_voca = None, embedding_file = None):

        super(GGIPNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Load pre-trained gene embeddings if specified
        if use_pre_trained_gene2vec:
            vocabulary = all_text_voca
            initW = None
            initW = NN_util.load_embedding_vectors(vocabulary, embedding_file, embedding_size)
            print("gene embedding file has been loaded\n")
            self.embedding.weight.data.copy_(torch.from_numpy(initW))
        else:
            print("embedding loading errors")

        # Fully connected layers
        self.fc1 = nn.Linear(sequence_length * embedding_size, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc3 = nn.Linear(hidden_dimension, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Flatten embedding
        flattened = embedded.view(embedded.size(0), -1)

        # Fully connected layers
        out = self.fc1(flattened)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)

        return out