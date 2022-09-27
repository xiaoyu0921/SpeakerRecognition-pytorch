import torch
import torch.nn as nn
from model.BaseSpeakerEncoder import BaseSpeakerEncoder
from model.TDNN import TDNN


class X_vector_Encoder(BaseSpeakerEncoder):
    def __init__(self, input_dim=40, num_classes=8, saved_model=""):
        super(X_vector_Encoder, self).__init__()
        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1, dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2, dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3, dropout_p=0.5)

        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 128)
        self.output = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def forward(self, inputs):
        # print("xvector_input: ", inputs.shape)
        tdnn1_out = self.tdnn1(inputs)
        # print("tdnn1_out:",tdnn1_out.shape)
        tdnn2_out = self.tdnn2(tdnn1_out)
        # print("tdnn2_out:",tdnn2_out.shape)
        tdnn3_out = self.tdnn3(tdnn2_out)
        # print("tdnn3_out:",tdnn3_out.shape)
        tdnn4_out = self.tdnn4(tdnn3_out)
        # print("tdnn4_out:",tdnn4_out.shape)
        tdnn5_out = self.tdnn5(tdnn4_out)
        # print("tdnn5_out:",tdnn5_out.shape)

        ### Stat Pool
        mean = torch.mean(tdnn5_out, 1)
        std = torch.std(tdnn5_out, 1)
        stat_pooling = torch.cat((mean, std), 1)
        # print("pool_out:", stat_pooling.shape)

        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)
        # print("x-vector:", x_vec.shape)

        predictions = self.softmax(self.output(x_vec))
        # print("predictions:", x_vec.shape)

        # return x_vec
        return predictions, x_vec