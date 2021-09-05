import warnings
warnings.filterwarnings('ignore')
import argparse
import re
from pathlib import Path

from tqdm.auto import tqdm


class PreProcessor:

    def __init__(self, args):

        self.label_dir = Path(args.label_dir)
        self.label_output_dir = Path(args.label_output_dir)
        self.label_output_dir.mkdir(parents=True, exist_ok=True)

    def process_label(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        phonemes, a1s, f2s = list(), list(), list()
        for line in lines:
            if line.split("-")[1].split("+")[0] == "pau":
                phonemes += ["pau"]
                a1s += ["xx"]
                f2s += ["xx"]
                continue
            p = re.findall(r"\-(.*?)\+.*?\/A:([+-]?\d+).*?\/F:.*?_([+-]?\d+)", line)
            if len(p) == 1:
                phonemes += [p[0][0]]
                a1s += [p[0][1]]
                f2s += [p[0][2]]
        assert len(phonemes) == len(a1s) and len(phonemes) == len(f2s)
        return phonemes, a1s, f2s

    def preprocess(self):
        labels = list()
        labels2 = list()
        label_paths = list(sorted(list(self.label_dir.glob('*.lab'))))
        for label_path in tqdm(label_paths[:550], total=len(label_paths)):
            phonemes, a1s, f2s = self.process_label(label_path)
            labels.append(f'JSUT/{label_path.name.replace(".lab", ".wav")}|0|{"_".join(phonemes)}|{"_".join(a1s)}|{"_".join(f2s)}\n')
            labels2.append(f'JMVD/msk_{label_path.name.replace(".lab", ".wav")}|1|{"_".join(phonemes)}|{"_".join(a1s)}|{"_".join(f2s)}\n')
        train = labels[:500] + labels2[:500]
        valid = labels[500:] + labels2[500:]
        with open(self.label_output_dir / 'train.txt', 'w', encoding='utf-8') as f:
            f.writelines(train)
        with open(self.label_output_dir / 'valid.txt', 'w', encoding='utf-8') as f:
            f.writelines(valid)


if __name__ == '__main__':
    try:
        import torchaudio
        torchaudio.set_audio_backend('sox_io')
    except RuntimeError:
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
        torchaudio.set_audio_backend('soundfile')
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--label_output_dir', type=str, default='./filelists')
    args = parser.parse_args()

    processor = PreProcessor(args)

    print('Start preprocessing')
    processor.preprocess()
