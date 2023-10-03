import glob
import os

LibriTTS_path = '/workspace/raid/dataset/LibriTTS/'

wavs= []
wavs += glob.glob(os.path.join(LibriTTS_path+'train-clean-100/*/*/*.wav'))
wavs += glob.glob(os.path.join(LibriTTS_path+'train-clean-360/*/*/*.wav'))


with open('train_libritts.txt', 'w') as f:
    for wav in wavs:
        f.write(wav+'\n')

wavs= []
wavs += glob.glob(os.path.join(LibriTTS_path+'dev-clean/*/*/*.wav'))

with open('dev_libritts.txt', 'w') as f:
    for wav in wavs:
        f.write(wav + '\n')

wavs= []
wavs += glob.glob(os.path.join(LibriTTS_path+'test-clean/*/*/*.wav'))

with open('test_libritts.txt', 'w') as f:
    for wav in wavs:
        f.write(wav + '\n')