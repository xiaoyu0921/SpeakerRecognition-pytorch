from model.BaseSpeakerEncoder import BaseSpeakerEncoder
import torch
import torch.nn as nn

from utils_ import myconfig


class LstmSpeakerEncoder(BaseSpeakerEncoder):

    # 初始化函数
    def __init__(self, saved_model=""):
        super(LstmSpeakerEncoder, self).__init__()
        # Define the LSTM network.
        self.lstm = nn.LSTM(
            input_size=myconfig.N_MFCC,
            hidden_size=myconfig.LSTM_HIDDEN_SIZE,
            num_layers=myconfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=myconfig.BI_LSTM)

        # Load from a saved model if provided.
        if saved_model:
            self._load_from(saved_model)

    def _aggregate_frames(self, batch_output):
        """Aggregate output frames."""
        # 平均值池化
        if myconfig.FRAME_AGGREGATION_MEAN:
            return torch.mean(
                batch_output, dim=1, keepdim=False)
        # 最后帧
        else:
            return batch_output[:, -1, :]

    # 正向传播函数
    def forward(self, x):
        # LSTM是不是双向LSTM
        # print("x.shape:", x.shape)

        D = 2 if myconfig.BI_LSTM else 1

        # 初始状态h0,c0
        h0 = torch.zeros(
            D * myconfig.LSTM_NUM_LAYERS, x.shape[0],  myconfig.LSTM_HIDDEN_SIZE
        ).to(myconfig.DEVICE)

        c0 = torch.zeros(
            D * myconfig.LSTM_NUM_LAYERS, x.shape[0], myconfig.LSTM_HIDDEN_SIZE
        ).to(myconfig.DEVICE)

        # 根据初始状态(h0,c0)和输入x，返回输出y，和新的状态hn,cn
        y, (hn, cn) = self.lstm(x, (h0, c0))

        # print("y:", y.shape)
        # print("y_a:", self._aggregate_frames(y).shape)

        # 将输出y聚合为声纹嵌入码
        return self._aggregate_frames(y)