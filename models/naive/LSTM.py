"""
LSTM model.  (c) 2021 Georgia Tech

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

# from Machine_Translation import device
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cpu')
class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # four gates for LSTM, each gate has two pairs of weights ( W_X w/ input, hidden; W_H w/ hidden, hidden ) and one bias.

        # Input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate (g_t)
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # Activations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""
        # get the batch size and sequence length, and initialize hidden state as well as cell state
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device = x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device = x.device)

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################

        for t in range(seq_len):
            # current time step input
            x_t = x[:, t, :]
            # input gate
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            # forget gate 
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            # cell gate
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            # output gate
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)


            # update cell state
            c_t = f_t * c_t + i_t * g_t
            # update hidden state
            h_t = o_t * torch.tanh(c_t)
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)

if __name__ == "__main__":
    lstm = LSTM(2, 5)
    x = torch.randn(4,3,2)
    h_t, c_t = lstm.forward(x)
    print(h_t.shape, c_t.shape)