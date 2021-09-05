import torch
from torch.nn.utils.rnn import pad_sequence


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, a1s, f2s]
        """
        phoneme, a1, f2, mel, spk_id = tuple(zip(*batch))

        phoneme_padded = pad_sequence(phoneme, batch_first=True)
        a1_padded = pad_sequence(a1, batch_first=True)
        f2_padded = pad_sequence(f2, batch_first=True)
        input_lengths = torch.LongTensor([len(x) for x in phoneme])

        mel_padded = pad_sequence(mel, batch_first=True).transpose(-1, -2)
        output_lengths = torch.LongTensor([len(x) for x in mel])

        spk_id = torch.LongTensor(spk_id)

        return phoneme_padded, a1_padded, f2_padded, input_lengths, mel_padded, output_lengths, spk_id
