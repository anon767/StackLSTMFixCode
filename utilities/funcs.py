import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
torch.autograd.set_detect_anomaly(True)

def train_one_epoch(training_loader, optimizer, model, loss_fn):
    last_loss = 0.

    for x_padded, y_padded, x_lens, y_lens in training_loader:

        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x_padded)
        str_predicts = model.predict(outputs)
        loss = loss_fn(outputs, y_padded.T)

        loss.backward(retain_graph=True)
        optimizer.step()   

        last_loss = last_loss + loss.item()

    return last_loss

def decode_predictions(y_hat, model):
    encoded_predictions = y_hat.argmax(axis=1).numpy()
    str_predictions_in_array = encoded_predictions, model
    str_predictions_in_lines = [' '.join(list(x)) for x in str_predictions_in_array]
    return str_predictions_in_lines

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  batch_sze = len(batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]

  padded_vectors = pad_sequence(xx + yy, padding_value=1)
  xx_pad,yy_pad = padded_vectors[:, :batch_sze], padded_vectors[:, batch_sze:]
  return xx_pad, yy_pad, x_lens, y_lens
