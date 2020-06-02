# How to install
* We use modified version of the official [Phonetisaurus G2P](https://github.com/AdolfVonKleist/Phonetisaurus)
* To run the script you need to install
    * mitlm
    * OpenFst-1.6.2 (must be 1.6.x version)
    * modified Phonetisaurus with Pyhon3 bindings

### Install prerequisties
```shell script
sudo apt-get update
# Basics
sudo apt-get install git g++ autoconf-archive make libtool
# Python bindings
sudo apt-get install python-setuptools python-dev
# mitlm
sudo apt-get install gfortran
```

### Install mitlm
```shell script
git clone https://github.com/mitlm/mitlm.git
cd mitlm/
./autogen.sh
make
sudo make install
cd
```

### Install Openfst 1.6.2 
```shell script
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz
tar -xvzf openfst-1.6.2.tar.gz
cd openfst-1.6.2

# Minimal configure, compatible with current defaults for Kaldi
./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
make -j 4

# Now wait a while...
sudo make install
cd

# Extend your LD_LIBRARY_PATH .bashrc:
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst' >> ~/.bashrc
source ~/.bashrc
```

### Install Modified Phonetisaurus + Python bindngs
```shell script
cd _extra_src/modified-phonetisaurus
./install.sh
```