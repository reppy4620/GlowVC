from .models import FlowGenerator
from .loss import mle_loss, duration_loss
from utils import get_hparams_from_dir, latest_checkpoint_path, load_checkpoint
from text import Tokenizer

N_VOCAB = len(Tokenizer())


def load_glow_tts(model_dir):
    hps = get_hparams_from_dir(model_dir)
    checkpoint_path = latest_checkpoint_path(model_dir)
    model = FlowGenerator(
        n_vocab=N_VOCAB,
        out_channels=hps.data.n_mel_channels,
        **hps.model
    )
    load_checkpoint(checkpoint_path, model)
    return model

