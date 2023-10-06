import torch
import numpy as np
import GGIPNN as GGIPNN
import GGIPNN_util as NN_util
import random
import os
import time
import datetime
from sklearn import metrics

# Parameters
# ==================================================

# Data loading params
embedding_file = "../pre_trained_emb/gene2vec_dim_200_iter_9.txt"

# Model Hyperparameters
l2_reg_lambda = 0
embedding_dimension = 200
dropout_keep_prob = 0.5

# Training parameters
batch_size = 128
num_epochs = 10
learning_rate = 0.001
evaluate_every = 200
checkpoint_every = 1000
num_checkpoints = 5

# Misc Parameters
# allow_soft_placement = True # TensorFlow can move operations to a different device 
# log_device_placement = False
use_pre_trained_gene2vec = False  # if False, the embedding layer will be initialized randomly
train_embedding = True  # if True, the embedding layer will be trained during the training

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


# Convert data to PyTorch tensors
x_train = torch.LongTensor(np.array(x_train))
y_train = torch.FloatTensor(np.array(y_train))
x_dev = torch.LongTensor(np.array(x_dev))
y_dev = torch.FloatTensor(np.array(y_dev))
x_test = torch.LongTensor(np.array(x_test))
y_test = torch.FloatTensor(np.array(y_test))

# Model Training
# ==================================================

# Initialize model
model = GGIPNN.GGIPNN(
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=len(all_text_voca),
    embedding_size=embedding_dimension,
    hidden_dimension=100,
    embedTrain=train_embedding,
    l2_lambda=l2_reg_lambda
)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(x_train) // batch_size + 1
for epoch in range(num_epochs):
    for i in range(total_step):
        # Get batch of data
        start = i * batch_size
        end = min((i + 1) * batch_size, len(x_train))
        batch_x = x_train[start:end]
        batch_y = y_train[start:end]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i + 1) % evaluate_every == 0:
            # Evaluate model on validation set
            with torch.no_grad():
                dev_preds = model(x_dev)
                dev_loss = criterion(dev_preds, y_dev)
                dev_auc = metrics.roc_auc_score(y_dev, dev_preds)
                print("Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Dev Loss: {:.4f}, Dev AUC: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step,
                          loss.item(), dev_loss.item(), dev_auc))

        # Save the model checkpoint
        # print(f"i: {i+1}  ckp: {checkpoint_every} res: {(i + 1) % checkpoint_every == 0}")
        if (i + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join("checkpoints", "model_epoch{}_step{}.ckpt".format(epoch + 1, i + 1))
            # print(f"{checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)

# Test the model
# ==================================================

# Load the best checkpoint
# best_checkpoint = torch.load(os.path.join("checkpoints", "best_model.ckpt"))
# model.load_state_dict(best_checkpoint)

# Evaluate the model on test set
with torch.no_grad():
    test_preds = model(x_test)
    test_loss = criterion(test_preds, y_test)
    test_auc = metrics.roc_auc_score(y_test, test_preds)

print("Test Loss: {:.4f}, Test AUC: {:.4f}".format(test_loss.item(), test_auc))