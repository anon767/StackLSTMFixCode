import torch
from torch.utils.data import DataLoader
import numpy as np
from models import NVDDataset, StackLSTM
from utilities import funcs
import logging

HIDDEN_SIZE_CONTROLLER = 6
EMBED_DIM = 5
HIDDEN_SIZE_STACK = 4
LR = 0.001
BATCH_SIZE = 2
EPOCHS = 5
logging.basicConfig(filename='info.log', filemode='w', level=logging.INFO)

train_dataset = NVDDataset(data_path="dataset_preprocessed.pkl", vocab_path="vocab.pkl")
test_dataset = NVDDataset(data_path="dataset_preprocessed.pkl", vocab_path="vocab.pkl", train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=funcs.pad_collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=funcs.pad_collate)

model = StackLSTM(embedding_size=train_dataset.vocab_size, 
                  embedding_dim=EMBED_DIM,
                  hidden_size_controller=HIDDEN_SIZE_CONTROLLER,
                  hidden_size_stack=HIDDEN_SIZE_STACK, 
                  batch_size=BATCH_SIZE, 
                  label_encoder=train_dataset.le)
 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_vloss = 1_000_000_000_000

for epoch in range(EPOCHS):
    logging.info(f'EPOCH {epoch + 1}:')
    model.train(True)
    epoch_loss = funcs.train_one_epoch(training_loader=train_loader,
                                       optimizer=optimizer,
                                       model=model,
                                       loss_fn=loss_fn)
    running_vloss = 0.0
    model.train(False)

    for i, vdata in enumerate(test_loader):
        vinputs, vlabels, _, _ = vdata
        vlabels = vlabels.reshape(1,-1)
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels).detach()
    logging.info(f'LOSS train {epoch_loss} valid {vloss}')
    if vloss < best_vloss:
        best_vloss = vloss
        model_path = f'model_{epoch}'
        torch.save(model.state_dict(), model_path)
