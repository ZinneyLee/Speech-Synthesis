import os
import torch
import argparse
import json
from glob import glob
import tqdm
import numpy as np
from scipy.io.wavfile import write
import torchaudio
import utils

import torchaudio.transforms as T
from torch.nn import functional as F
from pesq import pesq
import torchcrepe

from pitch_periodicity import from_audio, p_p_F

from modules.vocoder.hifigan.data import MelSpectrogramFixed
from hifigan_ver3 import (
    SynthesizerTrn
)


h = None
device = None
def parse_filelist(filelist_path):
  with open(filelist_path, 'r') as f:
    filelist = [line.strip() for line in f.readlines()]
  return filelist

def inference(a):

    os.makedirs(a.output_dir, exist_ok=True)

    mel_fn = MelSpectrogramFixed(
        sample_rate=h.data.sampling_rate,
        n_fft=h.data.filter_length,
        win_length=h.data.win_length,
        hop_length=h.data.hop_length,
        f_min=h.data.mel_fmin,
        f_max=h.data.mel_fmax,
        n_mels=h.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda(device)
    resample = T.Resample(
        h.data.sampling_rate,
        16000,
        resampling_method="kaiser_window",
    ).cuda(device)

    threshold = torchcrepe.threshold.Hysteresis()

    net_g = SynthesizerTrn(
            h.data.n_mel_channels,
            h.train.segment_size // h.data.hop_length,
            **h.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(a.checkpoint_file, net_g, None)

    data_path = parse_filelist(a.filelist)

    print("data num: ", len(data_path))

    i = 0
    mel_error = 0
    pesq_wb = 0
    pesq_nb = 0

    pitch_total = 0
    periodicity_total = 0
    i_pitch = 0
    i_f1 = 0
    f1_total = 0

    for path in tqdm.tqdm(data_path, desc="synthesizing each utterance"):

        audio, sample_rate = torchaudio.load(path)
        # To ensure upsampling/downsampling will be processed in a right way for full signals

        p = (audio.shape[-1] // h.data.hop_length + 1) * h.data.hop_length - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
        audio = audio.cuda(device).squeeze(1)

        mel = mel_fn(audio)

        file_name = os.path.splitext(os.path.basename(path))[0]
        with torch.no_grad():

            audio_hat = net_g.infer(mel)
            mel_hat = mel_fn(audio_hat.squeeze(1))

            mel_error += F.l1_loss(mel, mel_hat).item()

            ref = resample(audio.squeeze())
            deg = resample(audio_hat.squeeze())

            ori_audio_len = audio.shape[-1]//300
            true_pitch, true_periodicity = from_audio(ref, ori_audio_len)
            fake_pitch, fake_periodicity = from_audio(deg, ori_audio_len)

            pitch, periodicity, f1 = p_p_F(threshold, true_pitch, true_periodicity, fake_pitch, fake_periodicity)

            pitch_total += pitch
            f1_total += f1

            periodicity_total += periodicity

            ref = ref.cpu().numpy()
            deg = deg.cpu().numpy()

            pesq_wb += pesq(16000, ref, deg, 'wb')
            pesq_nb += pesq(16000, ref, deg, 'nb')

            audio_hat = audio_hat.squeeze() / (torch.abs(audio_hat).max()) * 32767.0
            audio_hat = audio_hat.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, "{}.wav".format(file_name))
            write(output_file, h.data.sampling_rate, audio_hat)


        i +=1

    mel_error = mel_error/i
    pesq_wb = pesq_wb/i
    pesq_nb = pesq_nb/i
    pitch_total = pitch_total/i
    periodicity_total = periodicity_total/i
    f1_total = f1_total/i

    with open(os.path.join(a.output_dir, 'score_list.txt'), 'w') as f:
        f.write('Model: {}\nCheckpoint: {}\nDataset: {}\n\n'.format(a.model_name, a.step, os.path.splitext(os.path.basename(a.filelist))[0]))
        f.write('Mel L1 distance: {}\nPESQ Wide Band: {}\nPESQ Narrow Band {}\n'.format(mel_error, pesq_wb, pesq_nb))
        f.write('Pitch: {}\nPeriodicity: {}\nV/UV F1: {}\n '.format(pitch_total, periodicity_total, f1_total))



def main():
    print('Initializing Inference Process..')

    checkpoint = 'hifigan_mrd5_c'
    step = 1480000
    # checkpoint = 'hifigan_msd1_mpd5'
    # step = 1360000

    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='filelists/test_other.txt')
    parser.add_argument('--output_dir', default='inference/'+checkpoint+"_"+str(step)+"_test_other")
    parser.add_argument('--model_name', default=checkpoint)
    parser.add_argument('--step', default=step)
    parser.add_argument('--checkpoint_file', default='checkpoints/'+checkpoint+'/G_'+str(step)+'.pth')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')

    global h

    h = utils.get_hparams_from_file(config_file)

    torch.manual_seed(1234)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)

if __name__ == '__main__':
    main()