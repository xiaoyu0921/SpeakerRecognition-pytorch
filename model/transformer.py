from model.BaseSpeakerEncoder import BaseSpeakerEncoder
from utils_ import myconfig
import torch
import torch.nn as nn


class TransformerSpeakerEncoder(BaseSpeakerEncoder):

    def __init__(self, saved_model=""):
        super(TransformerSpeakerEncoder, self).__init__()
        # Define the Transformer network.
        self.linear_layer = nn.Linear(myconfig.N_MFCC, myconfig.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=myconfig.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=myconfig.TRANSFORMER_DIM, nhead=myconfig.TRANSFORMER_HEADS,
            batch_first=True),
            num_layers=1)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear_layer(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, myconfig.TRANSFORMER_DIM).to(
            myconfig.DEVICE)
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]
