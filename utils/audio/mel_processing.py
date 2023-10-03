import torch
import torchaudio

from librosa.filters import mel as librosa_mel_fn
from torchaudio.transforms import MelSpectrogram, Spectrogram

from utils.audio.vad import trim_long_silences
import matplotlib.pyplot as plt

def load_wav_to_torch(full_path, noramlize=False):
  data, sampling_rate = torchaudio.load(full_path, normalize=noramlize)
  return data.squeeze(0), sampling_rate

def load_audio_to_torch(audio_path, hop_length, training=False):
    audio, sample_rate = torchaudio.load(audio_path)  # appylying audio normalize
    # To ensure upsampling/downsampling will be processed in a right way for full signals
    '''
    if not training:
        p = (audio.shape[-1] // hop_length + 1) * hop_length - audio.shape[-1]
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
    '''
    return audio.squeeze(), sample_rate


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}


class SpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(SpectrogramFixed, self).__init__()
        self.torchaudio_backend = Spectrogram(**kwargs)

    def forward(self, x):
        outputs = self.torchaudio_backend(x)

        return outputs[..., :-1]


class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)


    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs[..., :-1]


def spectrogram_torch(wav, max_wav_value, n_fft, hop_size, win_size, center=True, use_slicing=False):
    """ Waveform to linear-spectrogram. """
    wav = wav / max_wav_value  # wav normalize
    wav = wav.unsqueeze(0)
    if torch.min(wav) < -1.:
        print('min value is ', torch.min(wav))
    if torch.max(wav) > 1.:
        print('max value is ', torch.max(wav))
    global hann_window
    dtype_device = str(wav.dtype) + '_' + str(wav.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=wav.dtype, device=wav.device)
    # Padding waveform
    if wav.shape[-1] % hop_size != 0 or not use_slicing:
        p = (wav.shape[-1] // hop_size + 1) * hop_size - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, p), mode='constant')
    # Linear-spectrogram
    spec = torch.stft(wav, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)[:, :, :-1]
    wav = wav.squeeze(0)[:spec.shape[-1]*hop_size]
    assert wav.shape[-1] == spec.shape[-1] * hop_size, f"| wav: {wav.shape}, spec: {spec.shape}"
    return wav, spec.squeeze(0).T


def spec_to_mel_torch(spec, fft_size, num_mels, sample_rate, fmin, fmax):
    """ Linear-spectrogram to mel-spectrogram. """
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(wav, fft_size, hop_size, win_length, num_mels, fmin, fmax, sample_rate, center=True,
                          use_slicing=False):
    """ Waveform to mel-spectrogram. ONLY USE FOR CHECKING SLICING WAVEFORM DURING TRAINING. """
    if torch.min(wav) < -1.:
        print('min value is ', torch.min(wav))
    if torch.max(wav) > 1.:
        print('max value is ', torch.max(wav))
    # Spectrogram process
    global hann_window, mel_basis
    dtype_device = str(wav.dtype) + '_' + str(wav.device)
    wnsize_dtype_device = str(win_length) + '_' + dtype_device
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    # Padding waveform
    if wav.shape[-1] % hop_size != 0 or not use_slicing:
        p = (wav.shape[-1] // hop_size + 1) * hop_size - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, p), mode='constant')
    # Linear-spectrogram
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=wav.dtype, device=wav.device)
    spec = torch.stft(wav, fft_size, hop_length=hop_size, win_length=win_length, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    # Mel-spectrogram
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = spectral_normalize_torch(mel)[:, :, :-1]
    assert wav.shape[-1] == mel.shape[2] * hop_size, f"| wav: {wav.shape}, mel: {mel.shape}"
    return mel


def torch_wav2spec(wav_fn, fft_size, hop_size, win_length, num_mels, fmin, fmax, sample_rate, center=True):
    """ Waveform to linear-spectrogram and mel-sepctrogram. """
    # Read wavform
    wav, _, sr = trim_long_silences(wav_fn, sample_rate)
    if sr != sample_rate:
        raise ValueError(f"{sr} SR doesn't match target {sample_rate} SR")
    wav = torch.from_numpy(wav).unsqueeze(0)
    if torch.min(wav) < -1.:
        print('min value is ', torch.min(wav))
    if torch.max(wav) > 1.:
        print('max value is ', torch.max(wav))

    # Padding waveform
    p = (wav.shape[-1] // hop_size + 1) * hop_size - wav.shape[-1]
    wav = torch.nn.functional.pad(wav, (0, p), mode='constant')
    
    # Linear-spectrogram
    spec = SpectrogramFixed(n_fft=fft_size, hop_length=hop_size, win_length=win_length, 
                            window_fn=torch.hann_window, center=center)
    spec = spec(wav)

    # Mel-spectrogram
    mel = MelSpectrogramFixed(sample_rate=sr, n_fft=fft_size, win_length=win_length, hop_length=hop_size, 
                              f_min=fmin, f_max=fmax, n_mels=num_mels, window_fn=torch.hann_window)
    mel = mel(wav)

    # Wav-processing
    wav = wav.squeeze(0)[:mel.shape[2] * hop_size]

    assert wav.shape[-1] == mel.shape[-1] * hop_size, f"| wav: {wav.shape}, spec: {spec.shape}, mel: {mel.shape}"
    assert mel.shape[-1] == spec.shape[-1], f"| wav: {wav.shape}, spec: {spec.shape}, mel: {mel.shape}"
    return {"wav": wav.cpu().detach().numpy(), "linear": spec.squeeze(0).T.cpu().detach().numpy(),
            "mel": mel.squeeze(0).T.cpu().detach().numpy()}

def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return