import time
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing
from data_prepare import feature_extraction
from datasets import dataset
from model.x_vector import X_vector_Encoder
from model.Lstm import LstmSpeakerEncoder
from model.transformer import TransformerSpeakerEncoder
from utils_ import myconfig
from utils_.loss import get_triplet_loss_from_batch_output
from utils_.save_model import save_model


def get_speaker_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    if myconfig.USE_TDNN:
        return  X_vector_Encoder(saved_model=load_from).to(myconfig.DEVICE)
    if myconfig.USE_TRANSFORMER:
        return TransformerSpeakerEncoder(load_from).to(myconfig.DEVICE)
    else:
        return LstmSpeakerEncoder(load_from).to(myconfig.DEVICE)


def train_network(spk_to_utts, num_steps, saved_model=None, pool=None):
    start_time = time.time()
    # 用来记录损失函数的列表
    losses = []
    # 1、创建神经网络的函数（transformer or lstm）
    encoder = get_speaker_encoder()

    # 2、创建Adam优化器
    optimizer = optim.Adam(encoder.parameters(), lr=myconfig.LEARNING_RATE)
    print("Start training")

    # 3、根据训练步数进行主循环
    for step in range(num_steps):
        # 梯度清零
        optimizer.zero_grad()

        # 4、从spk_to_utts的字典中构建一个训练数据的批次
        '''
        myconfig.BATCH_SIZE：批次的大小
        pool：多进程的poll
        batch_input：接收返回的该批次训练数据的张量
        '''

        # batch_input.shape: (24, 100, 40)
        batch_input = feature_extraction.get_batched_triplet_input(
            spk_to_utts, myconfig.BATCH_SIZE, pool).to(myconfig.DEVICE)
        # print("input: ", batch_input.shape)

        # 5、使用训练数据批次来调用声纹编码器神经网络的推理，返回数据中所有utt的嵌入码
        # batch_output.shape:（24，128）
        if myconfig.USE_TDNN:
            batch_output, x_vector = encoder(batch_input)
        else:
            batch_output = encoder(batch_input)
        # print(batch_output.shape)

        # 6、计算三元损失函数
        loss = get_triplet_loss_from_batch_output(
            batch_output, myconfig.BATCH_SIZE)

        # 7、通过反向传播来计算损失函数的梯度
        loss.backward()

        # 8、根据计算的梯度来优化神经网络的参数
        optimizer.step()
        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if (saved_model is not None and
                (step + 1) % myconfig.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = saved_model
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint += ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint,
                       encoder, losses, start_time)

    training_time = time.time() - start_time
    print("Finished training in", training_time, "seconds")
    if saved_model is not None:
        save_model(saved_model, encoder, losses, start_time)
    return losses


def run_training():

    # 构建数据集，得到spk_to_utts的字典结构
    if myconfig.TRAIN_DATA_CSV:
        spk_to_utts = dataset.get_csv_spk_to_utts(
            myconfig.TRAIN_DATA_CSV)
        print("Training data:", myconfig.TRAIN_DATA_CSV)
    else:
        spk_to_utts = dataset.get_librispeech_spk_to_utts(
            myconfig.TRAIN_DATA_DIR)
        print("Training data:", myconfig.TRAIN_DATA_DIR)

    # 调用multiprocess.Pool 通过多进程来更好的利用电脑上的多个CPU核，使模型训练高效
    with multiprocessing.Pool(myconfig.NUM_PROCESSES) as pool:
        losses = train_network(spk_to_utts,
                               myconfig.TRAINING_STEPS,
                               myconfig.SAVED_MODEL_PATH,
                               pool)  # 多进程的pool
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    run_training()
