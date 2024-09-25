import random
import sys
import traceback
import gc
from pathlib import Path
from pdb import set_trace as st

import matplotlib.pyplot as plt
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from data import EEG_Standard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# import os

# Create training and testing split of the data. We do not use validation in this tutorial.
# train_set = EEGFull(training=True)
# test_set = EEGFull(training=False)


train_blk_idx=0
# train_blk_idx=1
# blk_idx=1
# train_tone_idx=3
train_tone_idx=4
# train_tone_idx=1 # try short tones
# train_tone_idx=2 # try short tones
# tone_idx=4

# train_split_ratio=0.1 # how many samples to use for training
# train_split_ratio=0.2 # how many samples to use for training
# train_split_ratio=0.3 # how many samples to use for training
train_split_ratio=0.5 # how many samples to use for training
# train_split_ratio=0.7 # how many samples to use for training
# train_split_ratio=0.8 # how many samples to use for training
# train_split_ratio=0.9 # how many samples to use for training

print(f'train_blk_idx: {train_blk_idx}, train_tone_idx: {train_tone_idx}')

train_set = EEG_Standard(blk_idx=train_blk_idx, tone_idx=train_tone_idx, training=True, train_split_ratio=train_split_ratio)

test_blk_idx = train_blk_idx
test_tone_idx = train_tone_idx

# test_blk_idx = 1 - train_blk_idx
# test_tone_idx = 7 - train_tone_idx
# test_tone_idx = train_tone_idx

print(f'test_blk_idx: {test_blk_idx}, test_tone_idx: {test_tone_idx}')

test_set = EEG_Standard(blk_idx=test_blk_idx, tone_idx=test_tone_idx, training=False, train_split_ratio=train_split_ratio)

print(f'train dataset size: {len(train_set)}, test set size: {len(test_set)}')

waveform, label = train_set[
    0]  # 1*358*64, 1/2/3/4, cntl/tinn, 0/1


print("Shape of waveform: {}".format(waveform.shape))  # 1, 16k

batch_size = 32

if device == "cuda":
    num_workers = 2
    # num_workers = 0 # TODO
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
# test_loader = train_loader
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    # collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

######################################################################
# Define the Network
# ------------------
#
# For this tutorial we will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture
# described in `this paper <https://arxiv.org/pdf/1610.00087.pdf>`__. An
# important aspect of models processing raw audio data is the receptive
# field of their first layer’s filters. Our model’s first filter is length
# 80 so when processing audio sampled at 8kHz the receptive field is
# around 10ms (and at 4kHz, around 20 ms). This size is similar to speech
# processing applications that often use receptive fields ranging from
# 20ms to 40ms.
#


class M5(nn.Module):

    def __init__(self, n_input=1, n_output=35, stride=1, n_channel=32):
        super().__init__()
        # self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.conv1 = nn.Conv1d(
            n_input, n_channel, kernel_size=16,
            stride=stride)  # 358 / 20 ~= 18, receptive field = 20ms
        self.bn1 = nn.BatchNorm1d(n_channel)
        # self.pool1 = nn.MaxPool1d(4)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        # self.pool2 = nn.MaxPool1d(4)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        # self.pool3 = nn.MaxPool1d(4)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        # self.pool4 = nn.MaxPool1d(4)
        self.pool4 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


# TODO, tune the parameters
model = M5(n_input=waveform.shape[0], n_output=2,
           n_channel=32)  # binary classification
model.to(device)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


n = count_parameters(model)
print("Number of parameters: %s" % n)

######################################################################
# We will use the same optimization technique used in the paper, an Adam
# optimizer with weight decay set to 0.0001. At first, we will train with
# a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
# to 0.001 during training after 20 epochs.
#

# optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=5,
    gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

######################################################################
# Training and Testing the Network
# --------------------------------
#
# Now let’s define a training function that will feed our training data
# into the model and perform the backward pass and optimization steps. For
# training, the loss we will use is the negative log-likelihood. The
# network will then be tested after each epoch to see how the accuracy
# varies during the training.
#


def train(model, epoch, log_interval):
    model.train()
    # for batch_idx, (data, playvec, group, target) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data) # TODO?
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


######################################################################
# Now that we have a training function, we need to make one for testing
# the networks accuracy. We will set the model to ``eval()`` mode and then
# run inference on the test dataset. Calling ``eval()`` sets the training
# variable in all modules in the network to false. Certain layers like
# batch normalization and dropout layers behave differently during
# training so this step is crucial for getting correct results.
#


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        # data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)

    print(
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n"
    )


######################################################################
# Finally, we can train and test the network. We will train the network
# for ten epochs then reduce the learn rate and train for ten more epochs.
# The network will be tested after each epoch to see how the accuracy
# varies during the training.
#

log_interval = 20
# n_epoch = 10
n_epoch = 20

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

# The transform needs to live on the same device as the model and the data.
# transform = transform.to(device)
with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");

######################################################################
# The network should be more than 65% accurate on the test set after 2
# epochs, and 85% after 21 epochs. Let’s look at the last words in the
# train set, and see how the model did on it.
#

# def predict(tensor):
#     # Use the model to predict the label of the waveform
#     tensor = tensor.to(device)
#     tensor = transform(tensor)
#     tensor = model(tensor.unsqueeze(0))
#     tensor = get_likely_index(tensor)
#     tensor = index_to_label(tensor.squeeze())
#     return tensor

# waveform, sample_rate, utterance, *_ = train_set[-1]
# ipd.Audio(waveform.numpy(), rate=sample_rate)

# print(f"Expected: {utterance}. Predicted: {predict(waveform)}.")

######################################################################
# Let’s find an example that isn’t classified correctly, if there is one.
#

# for i, (waveform, sample_rate, utterance, *_) in enumerate(test_set):
#     output = predict(waveform)
#     if output != utterance:
#         ipd.Audio(waveform.numpy(), rate=sample_rate)
#         print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
#         break
# else:
#     print("All examples in this dataset were correctly classified!")
#     print("In this case, let's just look at the last data point")
#     ipd.Audio(waveform.numpy(), rate=sample_rate)
#     print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")

######################################################################
# Feel free to try with one of your own recordings of one of the labels!
# For example, using Colab, say “Go” while executing the cell below. This
# will record one second of audio and try to classify it.
#

# def record(seconds=1):

#     from google.colab import output as colab_output
#     from base64 import b64decode
#     from io import BytesIO
#     from pydub import AudioSegment

#     RECORD = (
#         b"const sleep  = time => new Promise(resolve => setTimeout(resolve, time))\n"
#         b"const b2text = blob => new Promise(resolve => {\n"
#         b"  const reader = new FileReader()\n"
#         b"  reader.onloadend = e => resolve(e.srcElement.result)\n"
#         b"  reader.readAsDataURL(blob)\n"
#         b"})\n"
#         b"var record = time => new Promise(async resolve => {\n"
#         b"  stream = await navigator.mediaDevices.getUserMedia({ audio: true })\n"
#         b"  recorder = new MediaRecorder(stream)\n"
#         b"  chunks = []\n"
#         b"  recorder.ondataavailable = e => chunks.push(e.data)\n"
#         b"  recorder.start()\n"
#         b"  await sleep(time)\n"
#         b"  recorder.onstop = async ()=>{\n"
#         b"    blob = new Blob(chunks)\n"
#         b"    text = await b2text(blob)\n"
#         b"    resolve(text)\n"
#         b"  }\n"
#         b"  recorder.stop()\n"
#         b"})"
#     )
#     RECORD = RECORD.decode("ascii")

#     print(f"Recording started for {seconds} seconds.")
#     display(ipd.Javascript(RECORD))
#     s = colab_output.eval_js("record(%d)" % (seconds * 1000))
#     print("Recording ended.")
#     b = b64decode(s.split(",")[1])

#     fileformat = "wav"
#     filename = f"_audio.{fileformat}"
#     AudioSegment.from_file(BytesIO(b)).export(filename, format=fileformat)
#     return torchaudio.load(filename)

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we used torchaudio to load a dataset and resample the
# signal. We have then defined a neural network that we trained to
# recognize a given command. There are also other data preprocessing
# methods, such as finding the mel frequency cepstral coefficients (MFCC),
# that can reduce the size of the dataset. This transform is also
# available in torchaudio as ``torchaudio.transforms.MFCC``.
#
