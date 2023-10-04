import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import random
import os
import time
import datetime
from GGIPNN import GGIPNN
import GGIPNN_util as NN_util

# Parameters
# ==================================================

# Data loading params
embedding_file = "../pre_trained_emb/gene2vec_dim_200_iter_9.txt"  # embedding file address, matrix txt file

# Model Hyperparameters
l2_reg_lambda = 0.0
embedding_dimension = 200
dropout_keep_prob = 0.5

# Training parameters
batch_size = 128
num_epochs = 1
evaluate_every = 200
checkpoint_every = 1000
num_checkpoints = 5

# Misc Parameters
allow_soft_placement = True
log_device_placement = False
use_pre_trained_gene2vec = False  # if False, the embedding layer will be initialized randomly
train_embedding = False  # if True, the embedding layer will be trained during the training

# Data loading data
x_train_raw_f = open("../predictionData/train_text.txt", 'r')
x_train_raw = x_train_raw_f.read().splitlines()
x_train_raw_f.close()
y_train_raw_f = open("../predictionData/train_label.txt", 'r')
y_train_raw = y_train_raw_f.read().splitlines()
y_train_raw_f.close()

x_test_raw_f = open("../predictionData/test_text.txt", 'r')
x_test_raw = x_test_raw_f.read().splitlines()
x_test_raw_f.close()
y_test_raw_f = open("../predictionData/test_label.txt", 'r')
y_test_raw = y_test_raw_f.read().splitlines()
y_test_raw_f.close()

x_valid_raw_f = open("../predictionData/valid_text.txt", 'r')
x_valid_raw = x_valid_raw_f.read().splitlines()
x_valid_raw_f.close()
y_valid_raw_f = open("../predictionData/valid_label.txt", 'r')
y_valid_raw = y_valid_raw_f.read().splitlines()
y_valid_raw_f.close()

all_text_voca = NN_util.myFitDict(x_train_raw + x_valid_raw + x_test_raw, 2)
all_text_unshuffled = NN_util.myFit(x_train_raw + x_valid_raw + x_test_raw, 2, all_text_voca)

x_train_len = len(x_train_raw)
x_valid_len = len(x_valid_raw)
x_test_len = len(x_test_raw)

x_train_unshuffled = all_text_unshuffled[:x_train_len]
x_valid_unshuffled = all_text_unshuffled[x_train_len:x_train_len + x_valid_len]
x_test_unshuffled = all_text_unshuffled[x_train_len + x_valid_len:]

total = x_train_raw + x_valid_raw + x_test_raw

# Randomly shuffle training data
random_indices = list(range(len(x_train_raw)))
random.shuffle(random_indices)
x_train = [x_train_unshuffled[i] for i in random_indices]
y_onehot = NN_util.oneHot(y_train_raw + y_valid_raw + y_test_raw)

y_train_onehot = y_onehot[:x_train_len]
y_train = [y_train_onehot[i] for i in random_indices]

x_dev = x_valid_unshuffled
y_dev = y_onehot[x_train_len:x_train_len + x_valid_len]

x_test = x_test_unshuffled
y_test = y_onehot[x_train_len + x_valid_len:]

predicationGS = y_test

print("total training size: " + str(len(y_train)))
print("total test size: " + str(len(y_test)))
print("training start!")

# Define the GGIPNN model
model = GGIPNN(
    sequence_length=len(x_train[0]),
    num_classes=len(y_train[0]),
    vocab_size=len(all_text_voca),
    embedding_size=embedding_dimension,
    l2_lambda=l2_reg_lambda
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train_step(x_batch, y_batch):
    model.train()
    optimizer.zero_grad()
    logits = model(x_batch)
    logits = logits[0]  # Extract the tensor from the tuple
    loss = criterion(logits, torch.argmax(y_batch, dim=1))
    loss.backward()
    optimizer.step()
    return loss.item()

def dev_step(x_batch, y_batch):
    model.eval()
    with torch.no_grad():
        logits = model(x_batch)
        logits = logits[0]  # Assuming logits is a tuple, extract the tensor
        loss = criterion(logits, torch.argmax(y_batch, dim=1))
        return loss.item()

best_dev_loss = float('inf')

for epoch in range(num_epochs):
    batches = list(NN_util.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs=1))
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        x_batch = torch.LongTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)
        train_loss = train_step(x_batch, y_batch)
        current_step = (epoch * len(batches)) + (epoch + 1)
        if current_step % evaluate_every == 0:
            x_dev_tensor = torch.LongTensor(x_dev)
            y_dev_tensor = torch.FloatTensor(y_dev)
            dev_loss = dev_step(x_dev_tensor, y_dev_tensor)
            print(f"Step {current_step}, Dev Loss: {dev_loss}")
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                torch.save(model.state_dict(), "best_model.pt")

# Load the best model
# model.load_state_dict(torch.load("best_model.pt"))

# Prediction
model.eval()
predictions_score_human_readable = []

for x_test_batch in NN_util.batch_iter(x_test, batch_size, num_epochs):
    x_test_batch_tensor = torch.LongTensor(x_test_batch)
    # x_test_batch_tensor = torch.FloatTensor(x_test_batch)  # Convert to FloatTensor
    with torch.no_grad():
        logits = model(x_test_batch_tensor)
    # batch_scores = [torch.softmax(logit, dim=1)[:, 1].tolist() for logit in logits]
    batch_scores = [torch.softmax(logit.float(), dim=0).tolist() for logit in logits]
    predictions_score_human_readable.extend(batch_scores)

predicationGS = np.argmax(predicationGS, axis=1)

yscore = predictions_score_human_readable
ytrue = predicationGS

# print("Shape of ytrue:", np.array(ytrue).shape)
# print("Shape of yscore:", np.array(yscore).shape)


print("-------------------")
print("AUC score")
print(metrics.roc_auc_score(np.array(ytrue), np.array(yscore)))
