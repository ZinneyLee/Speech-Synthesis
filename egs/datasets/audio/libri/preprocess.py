'''
from data_gen.tts.base_preprocess import BasePreprocessor
import os

class LibriPreprocess(BasePreprocessor):
    def meta_data(self):
        for n in os.listdir(self.raw_data_dir):
            for m in os.listdir(self.raw_data_dir+'/'+n):
                for l in open(f'{self.raw_data_dir}/{n}/{m}/{n}_{m}.trans.tsv').readlines():
                    item_name, txt = l.strip().split('\t', maxsplit=1)
                    wav_fn = f"{self.raw_data_dir}/{n}/{m}/{item_name}.wav"
                    yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt}
'''


from data_gen.tts.base_preprocess import BasePreprocessor


class LibriPreprocess(BasePreprocessor):
    def meta_data(self):
        metadatas = open(f'{self.raw_data_dir}/metadata.txt').readlines()
        for l in metadatas:
            item_name, spk, txt, wav_fn = l.strip().split("|")
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk}

