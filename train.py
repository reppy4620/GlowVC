import warnings
warnings.filterwarnings('ignore')

import os
import torch
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modules import FlowGenerator, Adam, mle_loss, duration_loss
from data import TextMelDataset, TextMelCollate
from text import Tokenizer
from utils import (Tracker, get_hparams, get_logger, seed_everything,
                   load_checkpoint, save_checkpoint, save_dp, latest_checkpoint_path)


def train(epoch, model, optimizer, loader, logger, accelerator):
    model.train()
    tracker = Tracker()
    bar = tqdm(desc=f'Epoch: {epoch} ', total=len(loader))
    for phoneme, a1, f2, in_length, mel, out_length, spk_id in loader:
        optimizer.zero_grad()
        (z, z_m, z_logs, logdet, z_mask), _, (attn, logw, logw_) = model(
            phoneme, a1, f2, in_length, mel, out_length, g=spk_id, gen=False
        )
        l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_dur = duration_loss(logw, logw_, in_length)
        loss = l_mle + l_dur
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        bar.update()
        bar.set_postfix_str(f'Loss: {loss.item():.4f}, MLE: {l_mle.item():.4f}, Duration: {l_dur.item():.4f}')
        tracker.update(mle=l_mle.item(), dur=l_dur.item(), all=loss.item())
    logger.info(f'Train Epoch: {epoch}, '
                f'Loss: {tracker.all.mean():.6f}, '
                f'MLE Loss: {tracker.mle.mean():.6f}, '
                f'Duration Loss: {tracker.dur.mean():.6f}')
    bar.close()


def evaluate(epoch, model, loader, logger):
    model.eval()
    tracker = Tracker()
    with torch.no_grad():
        for phoneme, a1, f2, in_length, mel, out_length, spk_id in loader:
            (z, z_m, z_logs, logdet, z_mask), _, (attn, logw, logw_) = model(
                phoneme, a1, f2, in_length, mel, out_length, g=spk_id, gen=False
            )
            l_mle = mle_loss(z, z_m, z_logs, logdet, z_mask)
            l_dur = duration_loss(logw, logw_, in_length)
            loss = l_mle + l_dur
            tracker.update(mle=l_mle.item(), dur=l_dur.item(), all=loss.item())
    logger.info(f'Eval Epoch: {epoch}, '
                f'Loss: {tracker.all.mean():.6f}, '
                f'MLE Loss: {tracker.mle.mean():.6f}, '
                f'Duration Loss: {tracker.dur.mean():.6f}')
    return tracker.mle.mean(), tracker.dur.mean()


def main():
    hps = get_hparams()
    logger = get_logger(hps.model_dir)
    logger.info(hps)
    seed_everything(hps.train.seed)

    accelerator = Accelerator(fp16=True)
    print(accelerator.state)

    train_dataset = TextMelDataset(hps.data.train_file, hps.data)
    collate_fn = TextMelCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8
    )
    valid_dataset = TextMelDataset(hps.data.valid_file, hps.data)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8
    )

    model = FlowGenerator(
        n_vocab=len(Tokenizer()),
        out_channels=hps.data.n_mel_channels,
        **hps.model
    )
    optimizer = Adam(
        model.parameters(),
        dim_model=hps.model.hidden_channels,
        **hps.optimizer
    )
    model, optimizer._optim, train_loader, valid_loader = accelerator.prepare(
        model, optimizer._optim, train_loader, valid_loader
    )
    epochs = 1
    best_mle_loss = float('inf')
    best_dur_loss = float('inf')
    try:
        *_, best_losses, epochs = load_checkpoint(
            f'{hps.model_dir}/G_latest.pth',
            model,
            optimizer=optimizer
        )
        epochs += 1
        if best_losses is not None:
            best_mle_loss, best_dur_loss = best_losses
        optimizer.step_num = (epochs - 1) * len(train_loader)
        optimizer._update_learning_rate()
    except:
        logger.info('Start training from scratch')
        logger.info('Start initialization')
        for f in model.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)
        with torch.no_grad():
            for inputs in train_loader:
                _ = model(
                    *inputs, gen=False
                )
                break
        logger.info('End initialization')

    for epoch in range(epochs, hps.train.epochs+1):
        train(epoch, model, optimizer, train_loader, logger, accelerator)
        mle_loss_, dur_loss = evaluate(epoch, model, valid_loader, logger)
        if mle_loss_ < best_mle_loss:
            save_checkpoint(accelerator.unwrap_model(model), optimizer,
                            [best_mle_loss, best_dur_loss], epoch,
                            os.path.join(hps.model_dir, f'G_best.pth'))
            best_mle_loss = mle_loss_
        if dur_loss < best_dur_loss:
            save_dp(accelerator.unwrap_model(model), epoch,
                    os.path.join(hps.model_dir, f'DP_best.pth'))
            best_dur_loss = dur_loss
        save_checkpoint(accelerator.unwrap_model(model), optimizer,
                        [best_mle_loss, best_dur_loss], epoch,
                        os.path.join(hps.model_dir, f'G_latest.pth'))
        print('-'*70)


if __name__ == '__main__':
    main()
