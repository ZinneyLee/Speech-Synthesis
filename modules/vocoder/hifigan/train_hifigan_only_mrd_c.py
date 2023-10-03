import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import random

import commons
import utils

from data import AudioDataset, MelSpectrogramFixed
import torchaudio.transforms as T
from pesq import pesq

from hifigan_only_mrd_c import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss
)

torch.backends.cudnn.benchmark = True
global_step = 0

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 50000 + random.randint(0,100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).cuda(rank)

    train_dataset = AudioDataset(hps, training=True)
    train_sampler = DistributedSampler(train_dataset) if n_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset, batch_size=hps.train.batch_size, num_workers=8,
        sampler=train_sampler, drop_last=True
    )

    if rank == 0:
        test_dataset = AudioDataset(hps, training=False)
        eval_loader = DataLoader(test_dataset, batch_size=1)
        resample = T.Resample(
            hps.data.sampling_rate,
            16000,
            resampling_method="kaiser_window",
        ).cuda(rank)
    else:
        resample = None
    net_g = SynthesizerTrn(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model, rank=rank).cuda(rank)

    if rank == 0:
        num_param = get_param_num(net_g)
        print('Number of Parameters:', num_param)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])


    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g,
                                                   optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d,
                                                   optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, mel_fn, resample], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d, mel_fn, resample], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d, mel_fn, resample = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, y in enumerate(train_loader):
        y = y.cuda(rank, non_blocking=True)

        mel = mel_fn(y)
        with autocast(enabled=hps.train.fp16_run):

            y_hat = net_g(mel)

            with autocast(enabled=False):
                mel_hat = mel_fn(y_hat.squeeze())

            # Discriminator

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y.unsqueeze(1), y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y.unsqueeze(1), y_hat)
            with autocast(enabled=False):

                loss_mel = F.l1_loss(mel, mel_hat) * hps.train.c_mel

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update(
                    {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel})

                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                torch.cuda.empty_cache()
                evaluate(hps, net_g, mel_fn, resample, eval_loader, writer_eval)

                if global_step % hps.train.save_interval == 0:
                    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                    utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, mel_fn, resample,  eval_loader, writer_eval):
    generator.eval()
    val_err_tot = 0
    pesq_wb = 0
    pesq_nb = 0
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, y in enumerate(eval_loader):

            y = y.cuda(0)
            mel = mel_fn(y)
            y_hat= generator.module.infer(mel)
            y_hat_mel = mel_fn(y_hat.squeeze(1))

            val_err_tot += F.l1_loss(mel, y_hat_mel[:,:,:mel.size(-1)]).item()

            ref = resample(y.squeeze()).cpu().numpy()
            deg = resample(y_hat.squeeze()[:y.shape[1]]).cpu().numpy()

            pesq_wb += pesq(16000, ref, deg, 'wb')
            pesq_nb += pesq(16000, ref, deg, 'nb')
            if batch_idx > 500:
                break
            if batch_idx <=4:

                image_dict.update({
                    "gen/mel_{}".format(batch_idx): utils.plot_spectrogram_to_numpy(y_hat_mel.squeeze().cpu().numpy())
                })
                audio_dict.update({"gen/audio_{}".format(batch_idx): y_hat.squeeze()})

                if global_step == 0:
                    image_dict.update({"gt/mel_{}".format(batch_idx): utils.plot_spectrogram_to_numpy(mel.squeeze().cpu().numpy())})
                    audio_dict.update({"gt/audio_{}".format(batch_idx): y.squeeze()})

    val_err_tot /= batch_idx
    pesq_wb /= batch_idx
    pesq_nb /= batch_idx
    scalar_dict = {"val/mel": val_err_tot, "val/pesq_wb": pesq_wb, "val/pesq_nb": pesq_nb}
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )
    generator.train()


if __name__ == "__main__":
    main()
