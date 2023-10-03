import torch 
from torch.nn import functional as F
import math


import commons


def ped_loss(mel_hats, iter):
  loss = 0
  peds = []
  batch_size = mel_hats.size(0)//iter
  for i in range(iter-1):
    l = F.l1_loss(mel_hats[i*batch_size:(i+1)*batch_size].detach(), mel_hats[(i+1)*batch_size:(i+2)*batch_size])
    peds.append(l)
    loss += l

  return loss, peds

def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2

def refi_feature_loss(fmap_r, fmap_g, iter):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):

      rl = rl.repeat(iter, 1, 1, 1).float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

def kl_loss_for_porta(z_p, logs_q, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p, logs_p = torch.zeros_like(z_p), torch.zeros_like(z_p)
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

def CTC_loss(pho_hat, pho_target, spec_lengths, pho_target_lengths, z_mask=None, ls=False, weight=0.01):
  ctc_loss = F.ctc_loss(pho_hat, pho_target, spec_lengths, pho_target_lengths)

  if ls is True:
     #[T, B, 178])
    # pho_hat = pho_hat.permute(1,2,0) # B, 178, T
    # smoothing_target = torch.full_like(pho_hat, 1. / 178)
    #
    # # kl = ((pho_hat *(pho_hat/smoothing_target).log())*z_mask).sum() / pho_hat.shape[0]
    # kl = ((pho_hat *(pho_hat/smoothing_target).log())).sum() / pho_hat.shape[0]

    kl_inp = pho_hat.transpose(0, 1)
    kl_tar = torch.full_like(kl_inp, 1. / 178)
    kl = F.kl_div(kl_inp, kl_tar, reduction="batchmean")

    loss_with_LS = (1. - weight) * ctc_loss + weight * kl

    return loss_with_LS

  else:

    return ctc_loss

def mle_loss(z, m, logs, logdet, mask):
  z = z.float()
  m = m.float()
  logs = logs.float()
  logdet = logdet.float()
  mask = mask.float()

  l = torch.sum(logs) + 0.5 * torch.sum(torch.exp(-2 * logs) * ((z - m)**2)) # neg normal likelihood w/o the constant term
  l = l - torch.sum(logdet) # log jacobian determinant
  l = l / torch.sum(torch.ones_like(z) * mask) # averaging across batch, channel and time axes
  l = l + 0.5 * math.log(2 * math.pi) # add the remaining constant term
  return l