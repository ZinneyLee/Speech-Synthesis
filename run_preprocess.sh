export PYTHONPATH=.

# preprocess
python data_gen/tts/runs/preprocess.py --config egs/datasets/audio/libri/fs2_orig.yaml

# binarize
# python data_gen/tts/runs/binarize.py --config egs/datasets/audio/libri/fs2_orig.yaml