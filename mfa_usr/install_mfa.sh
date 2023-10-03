#!/bin/bash
set -e
pip uninstall -y typing
pip install --ignore-requires-python git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git@v2.0.0b3
# mfa version v2.0.0a14로 upgrade
mfa thirdparty download
sudo apt install -y libopenblas-base libsox-fmt-mp3 libfst8 libfst-tools