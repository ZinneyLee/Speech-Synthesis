# Multi-Speaker Speech Synthesis
## Project Introduction
**Objective**
The aim of this project is to develop a personalized speech synthesis model capable of generating speech in various voices. Many existing speech synthesis models are not robust to multi-speaker adaptation. Our goal is to create a model that can learn from and generate multiple voices, allowing users to synthesize speech in their preferred style.

**Motivation**
The motivation behind this project lies in the diverse applications of speech synthesis technology. Speech synthesis is used in voice assistants, speech interfaces, advertising voiceovers, virtual characters, and content creation, among others. By providing a variety of voices, users can personalize their speech messages or content with different styles and tones.

## Data Description (LibriTTS)
**LibriTTS** is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. The LibriTTS corpus is designed for TTS research. The main charactristics are listed below:
  1. The audio files are at 24kHz sampling rate.
  2. The speech is split at sentence breaks.
  3. Both original and normalized texts are included.
  4. Contextual information (e.g., neighbouring sentences) can be extracted.
  5. Utterances with significant background noise are excluded.

**Feature Structure**
```
FeaturesDict({
    'chapter_id': int64,
    'id': string,
    'speaker_id': int64,
    'speech': Audio(shape=(None,), dtype=int64),
    'text_normalized': Text(shape=(), dtype=string),
    'text_original': Text(shape=(), dtype=string),
})
```

You can download the dataset [here](https://www.openslr.org/60/). 


You should download all datasets in the form of *'???-clean.tar.gz'*. 


To download these, you need to have sufficient storage capacity of over 37GB.
  - If you want to use validation set, you can use *'dev-clean.tar.gz'*.
