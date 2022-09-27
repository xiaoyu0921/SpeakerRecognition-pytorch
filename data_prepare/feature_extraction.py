import librosa
import soundfile as sf
import random
import torch
import numpy as np

import datasets
from data_prepare import specaug
from datasets import dataset
from utils_ import myconfig


def extract_features(audio_file):
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = sf.read(audio_file)

    # print("waveform1: ", waveform.shape)

    # Convert to mono-channel.
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())
    # print("waveform2: ", waveform.shape)

    # Convert to 16kHz.
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)

    # print("waveform3: ", waveform.shape)

    features = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=myconfig.N_MFCC)

    # print("features: ", features.shape)

    # print("features_transpose: ", features.transpose().shape)
    # 做转置，从而符合pytorch的输入维度
    return features.transpose()


def extract_sliding_windows(features):
    """Extract sliding windows from features."""
    sliding_windows = []
    start = 0
    while start + myconfig.SEQ_LEN <= features.shape[0]:
        sliding_windows.append(features[start: start + myconfig.SEQ_LEN, :])
        start += myconfig.SLIDING_WINDOW_STEP
    return sliding_windows


def get_triplet_features(spk_to_utts):
    """Get a triplet of anchor/pos/neg features."""
    anchor_utt, pos_utt, neg_utt = dataset.get_triplet(spk_to_utts)
    return (extract_features(anchor_utt),
            extract_features(pos_utt),
            extract_features(neg_utt))


# 将特征裁剪成指定的序列长度，方便构建一个训练数据批次的张量
def trim_features(features, apply_specaug):
    """Trim features to SEQ_LEN."""
    full_length = features.shape[0]

    # 从当前特征序列中，随机选取一个长度为SEQ_LEN的窗口
    start = random.randint(0, full_length - myconfig.SEQ_LEN)
    trimmed_features = features[start: start + myconfig.SEQ_LEN, :]

    # 应用时频谱增强
    if apply_specaug:
        trimmed_features = specaug.apply_specaug(trimmed_features)

    return trimmed_features


# 用于多进程，可以让训练速度更快
class TrimmedTripletFeaturesFetcher:
    """The fetcher of trimmed features for multi-processing."""

    def __init__(self, spk_to_utts):
        self.spk_to_utts = spk_to_utts

    def __call__(self, _):
        """Get a triplet of trimmed anchor/pos/neg features."""
        anchor, pos, neg = get_triplet_features(self.spk_to_utts)

        # print(anchor.shape, pos.shape, neg.shape)

        # 查看每一个样本特征的维度，如果维度 < 窗口的大小，就重新采样
        while (anchor.shape[0] < myconfig.SEQ_LEN or
               pos.shape[0] < myconfig.SEQ_LEN or
               neg.shape[0] < myconfig.SEQ_LEN):
            anchor, pos, neg = get_triplet_features(self.spk_to_utts)

        # 将特征裁剪成指定的序列长度，方便构建一个训练数据批次的张量
        # stack：按照行将样本一行行对应，堆叠在一起
        batch_input = np.stack([trim_features(anchor, myconfig.SPECAUG_TRAINING),
                                trim_features(pos, myconfig.SPECAUG_TRAINING),
                                trim_features(neg, myconfig.SPECAUG_TRAINING)])

        return batch_input


# 获取一次训练过程中所用到的数据批次
def get_batched_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batched triplet input for PyTorch."""

    # 创建一个获取一个batch的三元组特征，运行__call__函数
    fetcher = TrimmedTripletFeaturesFetcher(spk_to_utts)

    # 判断是否使用多进程
    if pool is None:
        input_arrays = list(map(fetcher, range(batch_size)))
    else:
        """
            map()函数。需要传递两个参数，
            第一个参数就是需要引用的函数，
            第二个参数是一个可迭代对象，
            它会把需要迭代的元素一个个的传入第一个参数我们的函数中。因为我们的map会自动将数据作为参数传进去
        """
        input_arrays = pool.map(fetcher, range(batch_size))
        # print(len(input_arrays))
    # numpy to tensor
    # print(input_arrays[0].shape)
    batch_input = torch.from_numpy(np.concatenate(input_arrays)).float()
    # print(batch_input.shape)
    return batch_input