import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  (c) 2021 Georgia Tech

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
import torch
import torch.nn as nn
import torch.optim as optim


# import custom models
device = torch.device('cpu')

class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden being fed into the decoder           #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################
        # encoder
        encoder_out = self.encoder(source)

        if self.encoder.model_type == "LSTM":
            encoder_outputs, (h_n, c_n) = encoder_out

            # take last layer
            # h_n = h_n[-1].unsqueeze(0)  # take last layer
            # c_n = c_n[-1].unsqueeze(0)
            hidden_decoder = (h_n, c_n)
        
        else:  # RNN
            encoder_outputs, h_n = encoder_out
            # h_n = h_n.squeeze(0).unsqueeze(0)  # shape (1, batch, hidden)
            hidden_decoder = h_n
            # print("seq2seq.py RNN h_n:", h_n.shape)


        # 2) initialize decoder input (<sos>)
        input_decoder = source[:, 0].unsqueeze(1)    #(1, batch, hidden)

        # 3) initialize outputs tensor
        outputs = torch.zeros(batch_size, seq_len, self.decoder.output_size, device=self.device)

        # 2) decoder
        output, hidden = self.decoder(input_decoder, hidden_decoder, encoder_outputs)
        outputs[:, 0, :] = output

        for t in range(1, seq_len):
            input_decoder = output.argmax(dim=1).unsqueeze(1)  #(1, batch, hidden)
            # print("Before decoder:", 
            # hidden_decoder[0].shape if isinstance(hidden_decoder, tuple) else hidden_decoder.shape)

            output, hidden = self.decoder(input_decoder, hidden, encoder_outputs)
            # print("After decoder:", 
            # hidden[0].shape if isinstance(hidden, tuple) else hidden.shape)
            outputs[:, t, :] = output
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
