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
from text import Tokenizer
from transform import TacotronSTFT

noise_scale = .667
length_scale = 1.0
MAX_WAV_VALUE = 32768.0
sr = 24000


if __name__ == '__main__':
    file_path = 'filelists/test.txt'

    parser = ArgumentParser()
    parser.add_argument('--glow_tts', required=True, type=str)
    parser.add_argument('--hifi_gan', required=True, type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--play_audio', action='store_true')
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glow_tts = load_glow_tts(args.glow_tts)
    checkpoint_path = latest_checkpoint_path(args.glow_tts, regex='DP_best.pth')
    glow_tts.encoder.proj_w.load_state_dict(torch.load(checkpoint_path)['dp'])
    hifi_gan = load_hifi_gan(args.hifi_gan)
    glow_tts, hifi_gan = glow_tts.eval().to(device), hifi_gan.eval().to(device)

    tokenizer = Tokenizer()
    stft = TacotronSTFT()

    if args.play_audio:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            frames_per_buffer=1024,
            output=True
        )

    def infer(inputs):
        phoneme, a1, f2 = tokenizer(*inputs)
        length = torch.tensor([phoneme.size(-1)], dtype=torch.long)
        phoneme, a1, f2 = phoneme.unsqueeze(0), a1.unsqueeze(0), f2.unsqueeze(0)
        phoneme, a1, f2, length = phoneme.to(device), a1.to(device), f2.to(device), length.to(device)
        with torch.no_grad():
            (mel, *_), *_ = glow_tts(phoneme, a1, f2, length,
                                     gen=True,
                                     noise_scale=noise_scale,
                                     length_scale=length_scale)
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
        gt_path, *inputs = line.strip().split('|')
        fn = gt_path.split('/')[-1]
        if hasattr(args, 'gt_root'):
            gt_path = f'{gt_path}/{fn}'
        mel_gen, wav_gen = infer(inputs)
        d = output_dir / os.path.splitext(fn)[0]
        d.mkdir(exist_ok=True, parents=True)

        wav_gt, _ = torchaudio.load(gt_path)
        torchaudio.save(
            str(d / 'gt.wav'),
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

        if args.play_audio:
            stream.write((wav_gt.squeeze().numpy() * MAX_WAV_VALUE).astype(np.int16).tobytes())
            stream.write((wav_gen.squeeze().numpy() * MAX_WAV_VALUE).astype(np.int16).tobytes())

        gt_mel = stft.mel_spectrogram(wav_gt).squeeze()
        mel_gen = mel_gen.squeeze()
        save_mel(gt_mel, d / 'gt.png')
        save_mel(mel_gen, d / 'gen.png')
