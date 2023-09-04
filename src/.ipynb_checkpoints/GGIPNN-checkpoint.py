import torch
import torch.nn as nn
import torch.nn.functional as F

class GGIPNN(nn.Module):
    """
    A neural network for text classification.
    Uses an embedding layer, followed by a hidden and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, hidden_dimension=100, embedTrain = False, l2_lambda =0):

        super(GGIPNN, self).__init__()

        # Placeholders for input, output and dropout
        self.input_x = torch.nn.Variable(torch.long, [None, sequence_length], name="input_x")
        self.input_y = torch.nn.Variable(torch.float, [None, num_classes], name="input_y")
        self.dropout_keep_prob = torch.nn.Variable(torch.float, name="dropout_keep_prob")

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedTrain:
            self.embedding.weight.requires_grad = True

        # Hidden layer
        self.W2 = nn.Linear(embedding_size * sequence_length, hidden_dimension)
        self.b2 = nn.Parameter(torch.ones([hidden_dimension]))

        self.W3 = nn.Linear(hidden_dimension, hidden_dimension)
        self.b3 = nn.Parameter(torch.ones([hidden_dimension]))

        self.W4 = nn.Linear(hidden_dimension, 10)
        self.b4 = nn.Parameter(torch.ones([10]))

        # Dropout
        self.dropout = nn.Dropout(1 - self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        self.scores = self.W4(F.relu(self.W3(F.relu(self.dropout(self.W2(self.embedding(self.input_x))))))) + self.b4
        self.softmax_scores = F.softmax(self.scores, dim=1)
        self.predictions = torch.argmax(self.scores, 1)

        # CalculateMean cross-entropy loss
        self.loss = F.cross_entropy(self.scores, self.input_y)
        l2_losses = torch.tensor([0.0]).to(self.loss.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_losses += torch.norm(param) * l2_lambda
        self.loss += l2_losses

        # Accuracy
        self.accuracy = torch.mean(torch.eq(self.predictions, self.input_y))

