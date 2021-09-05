import os.path
import warnings
warnings.filterwarnings('ignore')

import pyaudio
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from modules.glow_tts import load_glow_tts
from modules.hifi_gan import load_hifi_gan
from utils import latest_checkpoint_path
from transform import TacotronSTFT

noise_scale = .667
length_scale = 1.0
MAX_WAV_VALUE = 32768.0
sr = 24000


if __name__ == '__main__':
    file_path = 'filelists/valid.txt'

    parser = ArgumentParser()
    parser.add_argument('--glow_tts', required=True, type=str)
    parser.add_argument('--hifi_gan', required=True, type=str)
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glow_tts = load_glow_tts(args.glow_tts)
    glow_tts.encoder.proj_w.load_state_dict(torch.load(f'{args.glow_tts}/DP_best.pth')['dp'])
    hifi_gan = load_hifi_gan(args.hifi_gan)
    glow_tts, hifi_gan = glow_tts.eval().to(device), hifi_gan.eval().to(device)

    stft = TacotronSTFT()

    def infer(mel):
        length = torch.tensor([mel.size(-1)], dtype=torch.long)
        mel, length = mel.to(device), length.to(device)
        with torch.no_grad():
            mel = glow_tts.voice_conversion(mel, length)
            wav = hifi_gan(mel).squeeze(0)
        mel, wav = mel.cpu(), wav.cpu()
        return mel, wav

    def save_mel(mel, path):
        plt.figure(figsize=(10, 7))
        plt.imshow(mel, aspect='auto', origin='lower')
        plt.savefig(path)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, total=len(lines)):
        gt_path, spk_id, *inputs = line.strip().split('|')
        if spk_id == 0:
            continue
        fn = gt_path.split('/')[-1]
        if hasattr(args, 'gt_root'):
            gt_path = f'{gt_path}/{fn}'
        wav = torchaudio.load(gt_path)[0]
        mel = stft.mel_spectrogram(wav)
        mel_gen, wav_gen = infer(mel)
        d = output_dir / os.path.splitext(fn)[0]
        d.mkdir(exist_ok=True, parents=True)
        wav_gt, _ = torchaudio.load(gt_path)
        torchaudio.save(
            str(d / 'src.wav'),
            wav,
            sr,
            encoding='PCM_S',
            bits_per_sample=16
        )
        torchaudio.save(
            str(d / 'tgt.wav'),
            wav_gt,
            sr,
            encoding='PCM_S',
            bits_per_sample=16
        )
        torchaudio.save(
            str(d / 'gen.wav'),
            wav_gen,
            sr,
            encoding='PCM_S',
            bits_per_sample=16
        )

        gt_mel = stft.mel_spectrogram(wav_gt).squeeze()
        mel_gen = mel_gen.squeeze()
        save_mel(mel.squeeze(), d / 'src.png')
        save_mel(gt_mel, d / 'tgt.png')
        save_mel(mel_gen, d / 'gen.png')
