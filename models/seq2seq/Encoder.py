"""
S2S Encoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden states of the Encoder(namely, Linear - ReLU - Linear).    #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        # 1) Embedding layer
        self.embedding = nn.Embedding(self.input_size,self.emb_size)
        # 2) RNN or LSTM layer
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.encoder_hidden_size, batch_first=True)
        elif self.model_type == "LSTM":
            self.rnn = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True)
        else:
            raise ValueError("Unsupported model_type. Choose 'RNN' or 'LSTM'")

        # 3) Linear layers with ReLu and Tanh
        self.linear1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.activation_final = nn.Tanh()

        # 4) Dropout
        self.dropout = nn.Dropout(dropout)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #                                                                           #
        #       Do not apply any tanh (linear layers/Relu) for the cell state when  #
        #       model_type is LSTM before returning it.                             #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################
        # input: (batch, seq_len)

        # embedding + dropout
        embedding = self.embedding(input)
        # dropout certain signals. still output is (batch_size, seq len , embedding dim)
        embedding = self.dropout(embedding)

        # if debug:
        #     print("embedding shape:", embedding.shape)
        #     print("embedding values:", embedding)

        # RNN / LSTM
        if self.model_type == "RNN":
            # output: (batch, seq_len, hidden), hidden: (1, batch, hidden)
            output,hidden= self.rnn(embedding)
            # if debug:
            #     print("Raw hidden shape (RNN):", hidden.shape)
            #     print("Raw hidden values:", hidden)
            hidden = hidden.squeeze(0)

            # Linear + activation
            hidden = self.linear1(hidden)
            hidden = self.activation(hidden)
            hidden = self.linear2(hidden)
            hidden = self.activation_final(hidden)  # (batch, decoder_hidden_size)

        else:  # LSTM
            # output: (batch, seq_len, hidden)
            # hidden, cell: (num_layers=1, batch, hidden)
            output, (hidden, cell) = self.rnn(embedding)
            # if debug:
            #     print("Raw hidden shape (LSTM):", hidden.shape)
            #     print("Raw hidden values:", hidden)
            #     print("Raw cell shape:", cell.shape)
            #     print("Raw cell values:", cell)
            # take last layer only: (batch, hidden)
            hidden_last = hidden[-1]
            cell_last = cell[-1]

            # Linear + activation on hidden only
            hidden_transformed = self.linear1(hidden_last)
            hidden_transformed = self.activation(hidden_transformed)
            hidden_transformed = self.linear2(hidden_transformed)
            hidden_transformed = self.activation_final(hidden_transformed)
            
            # return tuple for LSTM
            hidden = (hidden_transformed, cell_last)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden



# import os, sys

# project_root = "/Users/wenwen/Downloads/CS 7643/A3/assignment3_NLP_spr26"
# os.chdir(project_root)

# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# print("Current directory:", os.getcwd())
# print("Python path[0]:", sys.path[0])

# import numpy as np
# from utils import unit_test_values, set_seed_nb

# set_seed_nb()
# i, n, h = 10, 4, 2

# encoder = Encoder(i, n, h, h)
# x_array = np.random.rand(5,2) * 10
# x = torch.LongTensor(x_array)
# out, hidden = encoder.forward(x)

# expected_out, expected_hidden = unit_test_values('encoder')


# if out is not None:
#     diff = expected_out - out
#     print('Close to out: ', expected_out.allclose(out, atol=1e-3))
#     print('Close to hidden: ', expected_hidden.allclose(hidden, atol=1e-3))
#     print('diff', diff)
# else:
#     print("NOT IMPLEMENTED")