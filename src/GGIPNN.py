import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the GGIPNN model
class GGIPNN(nn.Module):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_dimension=100, embedTrain=False, l2_lambda=0):
        super(GGIPNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if not embedTrain:
            self.embedding.weight.requires_grad = False

        # Hidden layers
        self.fc1 = nn.Linear(embedding_size * sequence_length, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, 100)
        self.fc3 = nn.Linear(100, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Embedding layer
        embedded_chars = self.embedding(x)

        # Flatten the embeddings
        embedded_chars_flat = embedded_chars.view(embedded_chars.size(0), -1)

        # Hidden layers with ReLU activation and dropout
        hidden1 = F.relu(self.fc1(embedded_chars_flat))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden2_drop = self.dropout(hidden2)

        # Output layer
        logits = self.fc3(hidden2_drop)

        # Softmax and predictions
        softmax_scores = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        return logits, softmax_scores, predictions

# Sample training loop
def train(model, x_train, y_train, batch_size, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            logits, _, _ = model(x_batch)
            loss = criterion(logits, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(x_train)}, Loss: {loss.item()}")

# Example usage
if __name__ == "__main__":
    # Sample data
    sequence_length = 50
    num_classes = 10
    vocab_size = 10000
    embedding_size = 200
    x_train = torch.randint(0, vocab_size, (1000, sequence_length))
    y_train = torch.randint(0, num_classes, (1000,))

    # Create and train the model
    model = GGIPNN(sequence_length, num_classes, vocab_size, embedding_size)
    train(model, x_train, y_train, batch_size=64, num_epochs=5)
