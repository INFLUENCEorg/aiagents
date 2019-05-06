#!/bin/sh
python3 setup.py sdist bdist_wheel
pip3 install dist/aiagents-0.1.tar.gz
