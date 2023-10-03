export PYTHONPATH=.

# training
#CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/datasets/audio/libri/fs2_orig.yaml --exp_name fastspeech2 --reset

# inference
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/datasets/audio/libri/fs2_orig.yaml --exp_name fastspeech2 --reset --infer
