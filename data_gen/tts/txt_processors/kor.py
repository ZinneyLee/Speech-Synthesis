# Based on https://github.com/keithito/tacotron
# and https://github.com/keonlee9420/Expressive-FastSpeech2
import json
import re

from g2pk import G2p  # Docker can't install G2p. Do not use this library in the docker.
# from g2p_en import G2p
from jamo import h2j

from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor, register_txt_processors
from utils.text.text_encoder import PUNCS


@register_txt_processors("ko")
class TextProcessor(BaseTxtProcessor):
    # G2pk settings
    g2p = G2p()
    # Dictionary settings
    dictionary = json.load(open("./data_gen/tts/txt_processors/dict/korean.json", "r"))
    num_checker = "([+-]?\d{1,3},\d{3}(?!\d)|[+-]?\d+)[\.]?\d*"
    PUNCS += ",\'\""

    @classmethod
    def preprocess_text(cls, text):
        # Normalize basic pattern
        text = text.strip()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z가-힣{cls.PUNCS}]", "", text)
        text = re.sub(f" ?([{cls.PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{cls.PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub('\(\d+일\)', '', text)
        text = re.sub('\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)', '', text)
        text = re.sub(f"([{cls.PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)

        # Normalize with prepared dictionaries
        text = cls.normalize_with_dictionary(text, cls.dictionary["etc_dict"])
        text = cls.normalize_english(text, cls.dictionary["eng_dict"])
        text = cls.normalize_upper(text, cls.dictionary["upper_dict"])

        # number to hanguel
        text = cls.normalize_number(text, cls.num_checker, cls.dictionary)

        return text

    @staticmethod
    def normalize_with_dictionary(text, dictionary):
        """ Check special korean pronounciation in dictionary """
        if any(key in text for key in dictionary.keys()):
            pattern = re.compile("|".join(re.escape(key) for key in dictionary.keys()))
            return pattern.sub(lambda x: dictionary[x.group()], text)
        else:
            return text
    
    @staticmethod
    def normalize_english(text, dictionary):
        """ Convert English to Korean pronounciation """
        def _eng_replace(w):
            word = w.group()
            if word in dict:
                return dictionary[word]
            else:
                return word
        text = re.sub("([A-Za-z]+)", _eng_replace, text)

        return text

    @staticmethod
    def normalize_upper(text, dictionary):
        """ Convert lower English to Upper English and Changing to Korean pronounciation"""
        def upper_replace(w):
            word = w.group()
            if all([char.isupper() for char in word]):
                return "".join(dictionary[char] for char in word)
            else:
                return word
        text = re.sub("[A-Za-z]+", upper_replace, text)

        return text

    @classmethod
    def normalize_number(cls, text, num_checker, dictionary):
        """ Convert Numbert to Korean pronounciation """
        text = cls.normalize_with_dictionary(text, dictionary["unit_dict"])
        text = re.sub(num_checker + dictionary["count_checker"],
                      lambda x: cls.num_to_hangeul(x, dictionary, True),
                      text)
        text = re.sub(num_checker,
                      lambda x: cls.num_to_hangeul(x, dictionary, False),
                      text)
        
        return text

    @staticmethod
    def num_to_hangeul(num_str, dictionary, is_count=False):
        """ Normalize number prounciation """
        zero_cnt = 0
        # Check Korean count unit
        if is_count:
            num_str, unit_str = num_str.group(1), num_str.group(2)
        else:
            num_str, unit_str = num_str.group(), ""
        # Remove decimal separator
        num_str = num_str.replace(",", "")

        if is_count and len(num_str) > 2:
            is_count = False
        
        if len(num_str) > 1 and num_str.startwith("0") and "." not in num_str:
            for n in num_str:
                zero_cnt += 1 if n == "0" else 0
            num_str = num_str[zero_cnt:]
        
        kor = ""
        if num_str != "":
            if num_str == "0":
                return "영 " + (unit_str if unit_str else "")
            # Split float number
            check_float = num_str.split(".")
            if len(check_float) == 2:
                digit_str, float_str = check_float
            elif len(check_float) >= 3:
                raise Exception(f"| Wrong number format: {num_str}")
            else:
                digit_str, float_str = check_float[0], None
            
            if is_count and float_str is not None:
                raise Exception(f"| 'is_count' and float number does not fit each other")
            
            # Check minus or plus symbol
            digit = int(digit_str)
            if digit_str.startswith("-") or digit_str.startswith("+"):
                digit, digit_str = abs(digit), str(abs(digit))
            
            size = len(str(digit))
            tmp = []
            for i, v in enumerate(digit_str, start=1):
                v = int(v)
                if v != 0:
                    if is_count:
                        tmp += dictionary["count_dict"][v]
                    else:
                        tmp += dictionary["num_dict"][str(v)]
                        if v == 1 and i != 1 and i != len(digit_str):
                            tmp = tmp[:-1]
                    tmp += dictionary["num_ten_dict"][(size - i) % 4]
                
                if (size - i) % 4 == 0 and len(tmp) != 0:
                    kor += "".join(tmp)
                    tmp = []
                    kor += dictionary["num_tenthousand_dict"][int((size - i) / 4)]
            
            if is_count:
                if kor.startswith("한") and len(kor) > 1:
                    kor = kor[1:]
                
                if any(word in kor for word in dictionary["count_tenth_dict"]):
                    kor = re.sub("|".join(dictionary["count_tenth_dict"].keys()),
                                 lambda x: dictionary["count_tenth_dict"][x.group()],
                                 kor)
            
            if not is_count and kor.startswith("일") and len(kor) > 1:
                kor = kor[1:]
            
            if float_str is not None and float_str != "":
                kor += "영" if kor == "" else ""
                kor += "쩜 "
                kor += re.sub("\d",
                              lambda x: dictionary["num_dict"][x.group()],
                              float_str)
            
            if num_str.startswith("+"):
                kor = "플러스 " + kor
            elif num_str.startswith("-"):
                kor = "마이너스 " + kor
            
            if zero_cnt > 0:
                kor = "공" * zero_cnt + kor
            
            return kor + unit_str

    @classmethod
    def process(cls, text, preprocess_args):
        text = cls.preprocess_text(text).strip()
        words = [word for word in text.split(" ") if word != ""]
        text_struct = [[w, []] for w in words]
        i_word = 0
        for w in words:
            phs = [ph for ph in h2j(cls.g2p(w)) if ph != ""]
            text_struct[i_word][1].extend(phs)
            i_word += 1
        text_struct = cls.postprocess(text_struct, preprocess_args)

        return text_struct, text