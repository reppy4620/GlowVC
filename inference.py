import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from pathlib import Path

import pyaudio
import torch
from scipy.io.wavfile import write

from modules.glow_tts import load_glow_tts
from modules.hifi_gan import load_hifi_gan
from utils import latest_checkpoint_path
from text import TokenizerForInfer

noise_scale = .667
length_scale = 1.0
MAX_WAV_VALUE = 32768.0
sr = 24000


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--glow_tts', required=True, type=str)
    parser.add_argument('--hifi_gan', required=True, type=str)
    parser.add_argument('--play_audio', action='store_true')
    parser.add_argument('--save_wav', action='store_true')
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()

    if args.save_wav:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glow_tts = load_glow_tts(args.glow_tts)
    checkpoint_path = latest_checkpoint_path(args.glow_tts, regex='DP_best.pth')
    glow_tts.encoder.proj_w.load_state_dict(torch.load(checkpoint_path)['dp'])
    hifi_gan = load_hifi_gan(args.hifi_gan)
    glow_tts, hifi_gan = glow_tts.eval().to(device), hifi_gan.eval().to(device)

    tokenizer = TokenizerForInfer()

    if args.play_audio:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            frames_per_buffer=1024,
            output=True
        )

    def infer(s):
        phoneme, a1, f2, length = tokenizer(s)
        phoneme, a1, f2, length = phoneme.to(device), a1.to(device), f2.to(device), length.to(device)
        with torch.no_grad():
            (mel, *_), *_ = glow_tts(phoneme, a1, f2, length,
                                     gen=True,
                                     noise_scale=noise_scale,
                                     length_scale=length_scale)
            wav = (hifi_gan(mel).squeeze() * MAX_WAV_VALUE).detach().cpu().numpy().astype('int16')
        return wav

    try:
        while True:
            s = input('文章を入力してください >> ')
            try:
                wav = infer(s)
            except:
                wav = infer('有効な文章を入力してください．')
            if args.save_wav:
                write(f'{str(output_dir)}/{s}.wav', sr, wav)
            if args.play_audio:
                stream.write(wav.tobytes())
    except KeyboardInterrupt:
        stream.close()
