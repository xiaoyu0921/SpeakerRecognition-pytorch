import re
import os
import glob
import sys
import pickle
import random
import numpy as np
import argparse
from python_speech_features import logfbank
import vad_ex
import webrtcvad
from progress.bar import Bar
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import wave
import logging
import math
import pandas as pd


class Preprocess():
    def __init__(self, train_dir, output_train_dir, data_type, segment_length, spectrogram_scale):
        # Set hparams
        self.train_dir = train_dir
        self.output_train_dir = output_train_dir
        self.data_type = data_type
        self.segment_length = segment_length
        self.spectrogram_scale = spectrogram_scale

    def preprocess_data(self):
        if self.data_type == "libri":
            path_list = [x for x in glob.iglob(
                self.train_dir.rstrip("/") + "/*/*/*.flac")]
        elif self.data_type == "vox1":
            path_list = [x for x in glob.iglob(
                self.train_dir.rstrip("/") + "/wav/*/*/*.wav")]
        elif self.data_type == 'vox2':
            path_list = [x for x in glob.iglob(
                self.train_dir.rstrip("/") + "/wav/*/*/*.m4a")]
        elif self.data_type == 'mit':
            path_list = [x for x in glob.iglob(
                self.train_dir.rstrip("/") + "/*/*/*.wav")]
        else:
            raise ValueError("data type not supported")

        bar = Bar("Processing", max=(len(path_list)),
                  fill='#', suffix='%(percent)d%%')

        print(len(path_list))
        lengths = []
        audio_names = []
        pickle_paths = []
        speakers = []
        # 对每个音频进行预处理
        for path in path_list:
            bar.next()
            # 去静音
            wav_arr, sample_rate = self.vad_process(path)
            # signal,sample_rate = librosa.load(path,16000)
            # wav_arr = self.vad_audio(signal,sample_rate)
            if sample_rate != 16000:
                print("sample rate do meet the requirement")
                exit()
            lengths.append(wav_arr.shape[0] / sample_rate)
            audio_names.append(path.split('\\')[-1])

            # padding 音频裁减
            wav_arr = self.cut_audio(wav_arr, sample_rate)

            # 提取特征并保存
            pickleName, label = self.create_pickle(path, wav_arr, sample_rate)
            pickle_paths.append(pickleName)
            speakers.append(label)

        data_dict = {
            'pickle_path': pickle_paths,
            'name': audio_names,
            'speaker': speakers,
            'length': lengths,
            'audio_path': path_list,
        }
        data = pd.DataFrame(data_dict)
        data.to_csv('./libri_test.csv', index=0)
        bar.finish()

    # 裁减音频
    def cut_audio(self, wav_arr, sample_rate):
        singal_len = int(self.segment_length * sample_rate)
        n_sample = wav_arr.shape[0]
        if n_sample < singal_len:
            wav_arr = np.hstack((wav_arr, np.zeros(singal_len - n_sample)))
        else:
            wav_arr = wav_arr[(n_sample - singal_len) // 2:(n_sample + singal_len) // 2]
        return wav_arr


    # VAD去静音
    def vad_process(self, path):
        # VAD Process

        # 读取音频
        if self.data_type == "vox1":
            audio, sample_rate = vad_ex.read_wave(path)
        elif self.data_type == "vox2":
            audio, sample_rate = vad_ex.read_m4a(path)
        elif self.data_type == "libri":
            audio, sample_rate = vad_ex.read_libri(path)
        elif self.data_type == 'mit':
            audio, sample_rate = vad_ex.read_libri(path)

        # 定义vad算法
        vad = webrtcvad.Vad(1)
        # print("audio:", audio)

        # 将PCM的语音数据，生成语音帧
        frames = vad_ex.frame_generator(30, audio, sample_rate)

        frames = list(frames)

        # 过滤静音帧
        segments = vad_ex.vad_collector(sample_rate, 30, 300, vad, frames)
        total_wav = b""
        for i, segment in enumerate(segments):
            total_wav += segment
        # Without writing, unpack total_wav into numpy [N,1] array
        wav_arr = np.frombuffer(total_wav, dtype=np.int16)
        # print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
        return wav_arr, sample_rate

    def plot_spectrogram(self, spec, ylabel):
        fig = plt.figure()
        heatmap = plt.pcolor(spec)
        fig.colorbar(mappable=heatmap)
        plt.xlabel('Time(s)')
        plt.ylabel(ylabel)
        plt.tight_layout()
        # plt.show()

    # 提取fbank特征
    def extract_feature(self, wav_arr, sample_rate, path):
        save_dict = {}
        logmel_feats = logfbank(
            wav_arr, samplerate=sample_rate, nfilt=self.spectrogram_scale)
        save_dict["LogMel_Features"] = logmel_feats
        # self.plot_spectrogram(logmel_feats.T,'Filter Banks')
        return save_dict

    # 写入pickle文件
    def create_pickle(self, path, wav_arr, sample_rate):

        if round((wav_arr.shape[0] / sample_rate), 1) >= self.segment_length:
            # 提取特征
            save_dict = self.extract_feature(wav_arr, sample_rate, path)

            if self.data_type == "vox1" or self.data_type == "vox2":
                data_id = "_".join(path.split("/")[-3:])
                save_dict["SpkId"] = path.split("/")[-3]
                save_dict["ClipId"] = path.split("/")[-2]
                save_dict["WavId"] = path.split("/")[-1]
                if self.data_type == "vox1":
                    pickle_f_name = data_id.replace("wav", "pickle")
                elif self.data_type == "vox2":
                    pickle_f_name = data_id.replace("m4a", "pickle")

            elif self.data_type == "libri":
                # data_id = "_".join(path.split("/")[-3:])
                # print("path", path)

                data_id = path.split("\\")[-1].replace("-", "_")  # 音频格式 5514_19192_0011.wav
                # print(data_id)
                pickle_f_name = data_id.replace("flac", "pickle")

            elif self.data_type == 'mit':
                data_id = "_".join(path.split("\\")[-2:])
                pickle_f_name = data_id.replace("wav", "pickle")

            if not os.path.exists(self.output_train_dir):
                os.mkdir(self.output_train_dir)
            with open(self.output_train_dir + "/" + pickle_f_name, "wb") as f:
                pickle.dump(save_dict, f, protocol=3)

            audio_label = os.path.basename(pickle_f_name).split("_")[0]
        else:
            print("wav length smaller than 1.6s: " + path)

        return os.path.join(self.output_train_dir, pickle_f_name), audio_label

    # 绘制音频
    def draw_waveform(self, wav_arr, sample_rate):
        plt.figure()
        librosa.display.waveplot(wav_arr, sample_rate)

    # 绘制语谱图
    def draw_spectrum(self, filename):
        f = wave.open(filename, 'rb')
        # 得到语音参数
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        # 得到的数据是字符串，需要将其转成int型
        strData = f.readframes(nframes)
        wavaData = np.fromstring(strData, dtype=np.int16)
        # 归一化
        wavaData = wavaData * 1.0 / max(abs(wavaData))
        # .T 表示转置
        wavaData = np.reshape(wavaData, [nframes, nchannels]).T
        f.close()
        # 绘制频谱
        plt.specgram(wavaData[0], Fs=framerate, scale_by_freq=True, sides='default')
        plt.ylabel('Frequency')
        plt.xlabel('Time(s)')
        # plt.show()

    def test_singleAudio(self, path):
        self.draw_spectrum(path)
        signal, sample_rate = librosa.load(path, 16000)
        self.draw_waveform(signal, sample_rate)  # 预处理前
        # 去静音
        # wav_arr = self.vad_audio(signal,sample_rate)
        wav_arr, sample_rate = self.vad_process(path)
        self.draw_waveform(wav_arr, sample_rate)  # 去静音后
        # plt.show()
        # exit()
        if sample_rate != 16000:
            print("sample rate do meet the requirement")
            exit()
        # padding 音频裁减
        wav_arr = self.cut_audio(wav_arr, sample_rate)
        # self.draw_waveform(wav_arr,sample_rate)  # 预处理后
        # self.draw_spectrum(wav_arr,sample_rate)
        # 提取特征并保存
        self.create_pickle(path, wav_arr, sample_rate)
        # self.draw_spectrum(wav_arr,sample_rate)

        plt.show()


def main():
    # timit
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/TIMIT_M/TIMIT/train/ --pk_dir=/home/qmh/Projects/Datasets/TIMIT_M/train/ --data_type=mit
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/TIMIT_M/TIMIT/test/ --pk_dir=/home/qmh/Projects/Datasets/TIMIT_M/TIMIT_OUTPUT/test/ --data_type=mit
    # libri    
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/LibriSpeech/train-clean-100/ --pk_dir=/home/qmh/Projects/Datasets/LibriSpeech_O/train-clean-100/ --data_type=libri
    # python preprocess.py --in_dir=/home/qmh/Projects/Datasets/LibriSpeech/test-clean/ --pk_dir=/home/qmh/Projects/Datasets/LibriSpeech_O/test-clean/ --data_type=libri
    train_dir = r"E:\天津大学\实验室\数据集\LibriSpeech\dev-clean"
    test_dir = r"E:\天津大学\实验室\数据集\LibriSpeech\test-clean"
    output_train_dir = r"E:\天津大学\实验室\数据集\LibriSpeech\lib-pre\train"
    output_test_dir = r"E:\天津大学\实验室\数据集\LibriSpeech\lib-pre\test"
    data_type = "libri"  # choices=["libri", "vox1", "vox2", "mit"]
    # segment length in seconds
    segment_length = 3.0
    # scale of the input spectrogram
    spectrogram_scale = 40

    # 初始化文件路径等参数
    preprocess = Preprocess(test_dir, output_test_dir, data_type, segment_length, spectrogram_scale)

    # 音频预处理
    preprocess.preprocess_data()

    # path = "/home/qmh/Projects/Datasets/TIMIT_M/TIMIT/train/dr4/fcag0/sx153.wav"
    # path = "/home/qmh/Projects/Datasets/LibriSpeech/train-clean-100/200/126784/200-126784-0025.wav"
    # preprocess.test_singleAudio(path)


if __name__ == "__main__":
    main()
