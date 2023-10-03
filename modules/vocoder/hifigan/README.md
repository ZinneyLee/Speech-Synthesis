# HiFi-GAN Vocoder

## Getting Started

### Pre-requisites
0. Pytorch >=3.11 and torchaudio >= 0.12 
1. Download datasets
    1. Download the VCTK dataset
    2. Download the LibriTTS dataset
```sh
mkdir dataset
cd dataset
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip -d VCTK-Corpus
cd ..
```
2. Install PESQ and torchcrepe for obejective evaluation
```sh
pip install pesq
pip install torchcrepe

```

### Preprocessing 
1. filelist generation

```sh
python data_list.py
```

https://github.com/speech-team-korea/Vocoder/blob/main/hifigan/data_list.py#L4

2. resample the audio dataset for your sampling rate 

```sh
import torchaudio.transforms as T

resample = T.Resample(ori_sr, target_sr, resampling_method="kaiser_window").cuda(rank) # 미리 선언하기 필수! (속도 차이)

resampled_audio = resample(audio)
```

### Checkpoint

### LibriTTS (Training: train-clean-360 and train-clean-100, Evaluation: dev-clean)

| Model |Discriminator |Checkpoint|Mel L1|PESQ(wb)|PESQ(nb)|Pitch|periodicity|V/UV F1|
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HiFi-GAN | MSD1, MPD5 | 1360k |0.2634|3.804|4.055|22.2893|0.0945|0.9592|
| HiFi-GAN | MRD3, MPD5  |  1370k |0.2337|3.868|4.086|21.1949|0.0933|0.9605|
| HiFi-GAN | cMRD3, MPD5 |  1370k |0.2393|3.860|4.083|20.9583|0.0911|0.9615|
| HiFi-GAN | cMRD5 | [1480k](https://drive.google.com/drive/folders/1VNbHKL9ZRn4B6XZbFRIO6IRTaAo21z8E?usp=sharing)| 0.1825 |3.899|4.115|18.4445|0.0892|0.9632|
| HiFi-GAN | cMRD5, DWD(sub-band4) |  1410k |0.2021|3.888|4.111|19.7590|0.0907|0.9620|

### LibriTTS (Training: train-clean-360 and train-clean-100, Evaluation: test-clean)

| Model |Discriminator |Checkpoint|Mel L1|PESQ(wb)|PESQ(nb)|Pitch|periodicity|V/UV F1|
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HiFi-GAN | MSD1, MPD5 | 1360k |0.2663|3.789|4.051|22.1083|0.0876|0.9630|
| HiFi-GAN | MRD3, MPD5  |  1370k |0.2366|3.861|4.086|20.2745|0.0841|0.9654|
| HiFi-GAN | cMRD3, MPD5 |  1370k |0.2428|3.847|4.082|20.7039|0.0844|0.9653|
| HiFi-GAN | cMRD5 | [1480k](https://drive.google.com/drive/folders/1VNbHKL9ZRn4B6XZbFRIO6IRTaAo21z8E?usp=sharing)| 0.1848 |3.892|4.115|18.4069|0.0821|0.9668|
| HiFi-GAN | cMRD5, DWD(sub-band4) |  1410k |0.2051|3.880|4.109|20.4593|0.0852|0.9665|

### LibriTTS (Training: train-clean-360 and train-clean-100, Evaluation: test-other)

| Model |Discriminator |Checkpoint|Mel L1|PESQ(wb)|PESQ(nb)|Pitch|periodicity|V/UV F1|
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HiFi-GAN | MSD1, MPD5 | 1360k |0.2674|3.6826|4.009|33.9553|0.1076|0.9466|
| HiFi-GAN | MRD3, MPD5  |  1370k |0.2394|3.7551|4.044|29.3802|0.1031|0.9496|
| HiFi-GAN | cMRD3, MPD5 |  1370k |0.2447|3.7479|4.041|26.4522|0.1022|0.9511|
| HiFi-GAN | cMRD5 | [1480k](https://drive.google.com/drive/folders/1VNbHKL9ZRn4B6XZbFRIO6IRTaAo21z8E?usp=sharing)| 0.1840 |3.830|4.095|23.9614|0.0967|0.9548|
| HiFi-GAN | cMRD5, DWD(sub-band4) |  1410k |0.2055|3.800|4.080|25.6886|0.1001|0.9531|

### Kor16k (audiobook43, nikl, KoreanAudiobook_w, selvas, estsoft)

| Model | Discriminator | Checkpoint | Mel L1 | PESQ(wb)| PESQ(nb)|
|------|:---:|:---:|:---:|:---:|:---:|
| HiFi-GAN | cMRD5 | [1750k](https://drive.google.com/drive/folders/1vQw9na0giyRg7RH7l4I4ro4ytCfg-bYw?usp=share_link) |0.1494|3.23|3.722|

### Emotion16k (VCTK, ESD, LibriTTS)

| Model |Discriminator |Checkpoint|Mel L1|PESQ(wb)|PESQ(nb)|Pitch|periodicity|V/UV F1|Evaluation|
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HiFi-GAN | cMRD5 | [2930k](https://works.do/x4Aom19) |0.1307|3.5608|3.8810|18.6673|0.0882|0.9629|dev-clean|
| HiFi-GAN | MSD1, MPD5 | [2170k](https://works.do/Fsz5dvR) |0.1760|3.5159|3.8625|20.3505|0.0937|0.9599|dev-clean|
| HiFi-GAN | cMRD5 | 2930k |0.1299|3.5104|3.8570|18.2366|0.0818|0.9663|test-clean|
| HiFi-GAN | MSD1, MPD5 | 2170k |0.1770|3.4751|3.8447|21.0014|0.0866|0.9639|test-clean|
| HiFi-GAN | cMRD5 | 2930k |0.1316|3.4810|3.8594|24.2242|0.0961|0.9534|test-other|
| HiFi-GAN | MSD1, MPD5 | 2170k |0.1820|3.3958|3.8148|31.2644|0.1044|0.9480|test-other|
| HiFi-GAN | cMRD5 | 2930k |0.1168|3.7319|3.9022|18.6589|0.0819|0.9637|ESD|
| HiFi-GAN | MSD1, MPD5 | 2170k |0.1547|3.6978|3.8736|20.6303|0.0893|0.9593|ESD|

### Inference

1. Checkpoint 다운로드 --> 폴더 위치 설정
2. Generator는 모든 Hifi-GAN 모델이 동일하기때문에 아무 model.py 이용하면 됨!
