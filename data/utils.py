import torchaudio


def load_data(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_wav(file_path):
    return torchaudio.load(file_path)
