#!/bin/bash

PYTHON=python3 ./configure --enable-python
make
sudo make install
cd python
cp ../.libs/Phonetisaurus.so .
sudo python3 setup.py install

