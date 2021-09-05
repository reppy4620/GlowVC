import torch
from torch.utils.data import Dataset

from text import Tokenizer
from transform import TacotronSTFT
from .utils import load_data, load_wav


class TextMelDataset(Dataset):

    def __init__(self, data_file_path, hparams):
        self.audiopaths_and_text = load_data(data_file_path)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.add_noise = hparams.add_noise
        self.add_blank = getattr(hparams, "add_blank", False)  # improved version
        self.tokenizer = Tokenizer()
        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        _, sr = load_wav(self.audiopaths_and_text[0][0])
        if sr != self.stft.sampling_rate:
            raise ValueError(f"{sr} {self.stft.sampling_rate} SR doesn't match target SR")

    def get_mel_text_pair(self, audiopath_and_text):
        file_path, spk_id, *text = audiopath_and_text
        text, a1, f2 = self.get_text(text)
        mel = self.get_mel(file_path)
        spk_id = int(spk_id)
        return text, a1, f2, mel, spk_id

    def get_mel(self, filename):
        wav, sr = load_wav(filename)
        if self.add_noise:
            wav = wav + torch.rand_like(wav) / self.max_wav_value
        mel = self.stft.mel_spectrogram(wav).squeeze().transpose(0, 1)
        return mel

    def get_text(self, text):
        phoneme, a1, f2 = self.tokenizer(*text)
        return phoneme, a1, f2

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
