from torch import nn 
from torch.utils.data import Dataset
import torch
import numpy as np

from stacknn.structs import Stack, buffers

import pickle 
from utilities import funcs
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import LabelEncoder

class StackLSTM(nn.Module):
     def __init__(self, 
                 embedding_size, 
                 embedding_dim,
                 hidden_size_controller,
                 hidden_size_stack,
                 batch_size, 
                 label_encoder) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hidden_size_controller = hidden_size_controller
        self.hidden_size_stack = hidden_size_stack
        self.label_encoder = label_encoder

        self.embedding = nn.Embedding(self.embedding_size, embedding_dim)

        self.controller = nn.LSTMCell(input_size=embedding_dim + hidden_size_stack, hidden_size=hidden_size_controller)

        self.output_linear = nn.Linear(in_features=hidden_size_controller, out_features=self.embedding_size)
        self.softmax = nn.Softmax()

        # self.input_buffer = buffers.InputBuffer(batch_size=batch_size, embedding_size=hidden_size_stack)
        # self.output_buffer = buffers.OutputBuffer(batch_size=batch_size, embedding_size=hidden_size_stack)

        self.push_fc = nn.Linear(hidden_size_controller, 1)
        self.pop_fc = nn.Linear(hidden_size_controller, 1)
        self.values_fc = nn.Linear(hidden_size_controller, hidden_size_stack)

        # self.input_fc = nn.Linear(hidden_size_controller, 1)    
        # self.output_fc = nn.Linear(hidden_size_controller, 1)

        self.sigmoid = nn.Sigmoid()

     def forward(self, x):
        self.stack = Stack(batch_size=self.batch_size, embedding_size=self.hidden_size_stack)
        embedded_x = self.embedding(x)
        hx, cx, rx = self.init_hidden()
        outputs = [] 
        for i, curr_x in enumerate(embedded_x):
            cat_x_rx = torch.cat((curr_x, rx), axis=-1)
            hx, cx = self.controller(cat_x_rx, (hx, cx))

            pop = self.sigmoid(self.pop_fc(hx))
            values = self.sigmoid(self.values_fc(hx))
            push = self.sigmoid(self.push_fc(hx))
            rx = self.stack(values, pop, push)
            outputs.append(self.output_linear(hx))
        return torch.stack(outputs, dim=2)

     def init_hidden(self):
        hx = torch.zeros((self.batch_size, self.hidden_size_controller)) # batch, hidden_size
        cx = torch.zeros((self.batch_size, self.hidden_size_controller)) 
        rx = torch.ones((self.batch_size, self.hidden_size_stack))
        # ox = torch.ones((self.batch_size, self.hidden_size_stack))
        # ix = torch.ones((self.batch_size, self.hidden_size_stack))

        return hx, cx, rx

     def predict(self, outputs):
        encoded_predictions = outputs.argmax(axis=1).numpy()
        translated_result = np.zeros_like(encoded_predictions).astype(object)
        for idx, seq in enumerate(encoded_predictions):
            translated_result[idx] = self.label_encoder.inverse_transform(seq)
        return translated_result

        


class NVDDataset(Dataset):
    def __init__(self, data_path, vocab_path, train=True, train_share=0.8):
        self.train_share = train_share
        self.test_share = 1 - train_share
        with open(vocab_path, "rb") as input_file:
            vocab = pickle.load(input_file)
        self.le = LabelEncoder()
        # start of code, end of code
        self.le.fit(["<SOC>", "<EOC>"])
        self.le.fit(list(vocab))
        self.vocab_size = len(self.le.classes_)

        with open(data_path, "rb") as input_file:
            self.data = pickle.load(input_file)
            if train:
                self.data = self.data[:int(len(self.data)*self.train_share)]
            else:
                self.data = self.data[-int(len(self.data)*self.test_share):]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vuln = torch.tensor(self.le.transform(self.data[index]["x"]))
        patch = torch.tensor(self.le.transform(self.data[index]["y"]))
        return vuln, patch
