# Training Marian NMT

This file describes how to train your own NMT system using the Marian toolkit.

## Getting to MetaCentrum and a GPU machine

NMT needs large-memory GPU cards (4 GB at least, better with 8 GB, or more GPUs
at once). Register at MetaCentrum to get access to a machine with such a card:

  https://metavo.metacentrum.cz/en/application/index.html

Clusters with GPU available:

- Doom machines have 5GB cards: https://metavo.metacentrum.cz/pbsmon2/resource/doom.metacentrum.cz
- Adan machines have 16GB cards: https://wiki.metacentrum.cz/wiki/Cluster_Adan

There are also smaller GPU clusters, see the full list here:
  https://wiki.metacentrum.cz/wiki/GPU_clusters

```
# First get to MetaCentrum
ssh nympha.zcu.cz  # or use another entry node

# Submit an interactive PBS job to get access to a GPU machine
qsub -q gpu -l select=1:ncpus=2:ngpus=1:mem=20gb:cl_doom=True \
     -l walltime=03:00 -I
  # !!! the -l flags says that your job will be killed after 3 minutes
  # all -q gpu jobs will be killed after 24 hours
  # use -q gpu_long and ask for up to 168 hours
  # you can ask for 2 gpus, always ask for 2x as many CPUs as GPUs
```

To see if you correctly landed on a GPU machine, see the usage of GPUs:
```
nvidia-smi
```

To see which GPUs your programs are allowed to use (as many as you asked for), check this:
```
echo $CUDA_VISIBLE_DEVICES
```

## Compiling Marian without Subword Units Support

This is the baseline compilation of Marian. See below for including SentencePiece.

```
ssh nympha.zcu.cz # or another MetaCentrum node
ssh doom7
  # or another free node, for doom visible here: https://metavo.metacentrum.cz/pbsmon2/resource/doom.metacentrum.
Note: For training never just ssh to the machine, always use PBS/qsub as shown above. For compilation, you may use ssh, but check if the machine is not already overloaded (run top).

# Activate prerequisites using MetaCentrum setup tools:
module add cmake-3.6.1
module add cuda-8.0
module add gcc-5.3.0
  # this particular GCC is needed for good compatibility with CUDA

# Make sure a tempdir with sufficient quota.
# CHECK YOUR QUOTAS on the web page:
#   https://metavo.metacentrum.cz/osobniv3/quotas/user/YOUR_METACENTRUM_USERNAME
#   (use your ssh username+password to log in to these pages)
export TMP=/SOME/PATH/WHERE/100GB/FIT

# Get Marian
git clone https://github.com/marian-nmt/marian.git

cd marian

# Compile Marian
#  (these are essentially instructions from https://marian-nmt.github.io/quickstart/)
mkdir build
cd build
cmake ..
make -j
```

It may happen, that the build process fails due to randomly occuring bugs regarding the disk space. If that happens, simply run the `make -j` command again. Only targets which weren't successful in the first run will be compiled. The second run should then be much faster.

### Compiling Marian with SentencePiece

Activate the following dependencies before compiling marian
(on your machine, you would `apt-get install` them):

```
module add protobuf-3.11.0
module add gperftools-2.7 
module add cmake-3.6.1
module add cuda-8.0
module add gcc-5.3.0
ppath=/software/protobuf/3.11.0/
```

Compile Marian:

```
git clone https://github.com/marian-nmt/marian marian_sentencesplitting
cd marian_sentencesplitting
mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SENTENCEPIECE=ON \
  -DPROTOBUF_LIBRARY=$ppath/lib/libprotobuf.so \
  -DPROTOBUF_INCLUDE_DIR=$ppath/include \
  -DPROTOBUF_PROTOC_EXECUTABLE=$ppath/bin/protoc \
  -DTCMALLOC_LIB=/software/gperftools/2.7/lib/libtcmalloc.so \
  -DCMAKE_INSTALL_PREFIX=../bin 
make -j 4

# verify the piecing is available
./marian --help |& grep sentencepiece
```
(Thanks to James Level for figuring this out.)


## Building Marian on AIC
### Without SentencePiece
```
# first go to AIC frontend
ssh username@aic.ufal.mff.cuni.cz

# then to some gpu node
qrsh -q gpu.q -l gpu=1,gpu_ram=8G,mem_free=16G,act_mem_free=16G,h_vmem=16G -pty yes bash -l

# it is better to add this to ~/.bashrc
export CUDA_HOME=/lnet/aic/opt/cuda/cuda-10.1

# newer cmake needed:
wget https://github.com/Kitware/CMake/releases/download/v3.14.2/cmake-3.14.2-Linux-x86_64.tar.gz
tar -zxvf cmake-3.14.2-Linux-x86_64.tar.gz
export CMAKE=cmake-3.14.2-Linux-x86_64/bin/cmake

# get marian (1.9.0 currently)
git clone https://github.com/marian-nmt/marian.git
mkdir marian/build && cd marian/build

# now make 
# turn off compilation for cpu (build fails otherwise)
$CMAKE .. -DCOMPILE_CPU=OFF
make -j4
```

## Getting Sample Training Data

There are two options, you are free to choose any of them with no impact on your homework score.

Dataset A is smaller and easier to train. BLEU scores on its corresponding testset will be higher, but the alignment as well as translation quality on our final czenali corpus will be lower.

Dataset B is realistic, you can get a very good English-Czech MT system with it. Aside from the (much) longer training time needed, you will also need to work with subword units, i.e. break words into short sequences of characters. The translation quality on our final czenali test set will be much higher and the alignment quality will be also higher, but you will have to deal with alignments predicted for subwords instead of the original tokens.

### A: Small and Simple Dataset
Use the training, development and test corpus from WMT Multimodal Task, as available from the Multi30k Github repository.

```
# Clone the dataset
git clone https://github.com/multi30k/dataset.git multi30k-dataset
We will use only the pre-processed, i.e. tokenized and lowercased, data as available in the directory:

multi30k-dataset/data/task1/tok:
  train.lc.norm.tok.en ... English source of training data
  train.lc.norm.tok.cs ... Czech target of training data
  val.lc.norm.tok.en ... English source of the development set
  val.lc.norm.tok.cs ... Czech reference translation of the development set
  test_2016_flickr.lc.norm.tok.en  ... English source of the test set called FLICKR in HW4 below
  test_2016_flickr.lc.norm.tok.cs  ... Czech reference translation for the test set
```

### B: Realistic and Better Dataset

This larger dataset requires you to use subword units, either SentencePiece compiled into Marian, or BPE run outside of Marian. The tutorial here shows BPE.

```
# Get the package
wget http://data.statmt.org/wmt17/nmt-training-task/wmt17-nmt-training-task-package.tgz
tar xzf wmt17-nmt-training-task-package.tgz

# Preprocess with BPE, i.e. break into pre-trained subwords
git clone https://github.com/rsennrich/subword-nmt

# Apply BPE to the training and development data
## You need to fix paths and do this for all files.
### This example shows it only for newstest2016, alias devset.
## The bpe_merges file is the same for both Czech and English.
zcat wmt17-nmt-training-task-package/newstest2016.en.gz \
| subword-nmt/subword_nmt/apply_bpe.py \
    --codes wmt17-nmt-training-task-package/bpe_merges \
    --input /dev/stdin --output /dev/stdout \
> dev.bpe.src

## When you do it for all the files you need, you will have:
##   train.bpe.src    ... English source of training data
##   train.bpe.tgt    ... Czech target of training data
##   dev.bpe.src      ... English source of the development set
##   dev.bpe.tgt      ... Czech target of the development set

# Start Training
# This is the baseline tested command to run interactively, if you asked for 2 GPUs:

marian \
  --devices 0 1  \
  --train-sets train.src train.tgt  \
  --model model.npz \
  --mini-batch-fit \
  --layer-normalization \
  --dropout-rnn 0.2 \
  --dropout-src 0.1 \
  --dropout-trg 0.1 \
  --early-stopping 5  \
  --save-freq 1000 \
  --disp-freq 1000
```

Make sure to use the correct training files depending on your dataset choice: `train.***.src` and `train.***.tgt`.

The above flags saved a model about every 5 minutes. In early stages of debugging, use `--save-freq` and `--disp-freq` of 100.

Use `--devices 0` if you asked for just 1 GPU.

For dataset B: add `--no-shuffle`, the dataset is already shuffled so no need to waste time on it.

Killing the command and starting it over will continue training from the last saved model.npz (but reading the corpus from the beginning).

The console will show things like:

```
[2017-12-05 15:03:02] [data] Loading vocabulary from train.src.yml
[2017-12-05 15:03:03] [data] Setting vocabulary size for input 0 to 43773
[2017-12-05 15:03:03] [data] Loading vocabulary from train.tgt.yml
[2017-12-05 15:03:03] [data] Setting vocabulary size for input 1 to 70560
[2017-12-05 15:03:03] [batching] Collecting statistics for batch fitting
[2017-12-05 15:03:06] [memory] Extending reserved space to 2048 MB (device 0)
[2017-12-05 15:03:06] [memory] Extending reserved space to 2048 MB (device 1)
[2017-12-05 15:03:06] [memory] Reserving 490 MB, device 0
[2017-12-05 15:03:08] [memory] Reserving 490 MB, device 0
[2017-12-05 15:03:37] [batching] Done
[2017-12-05 15:03:37] [memory] Extending reserved space to 2048 MB (device 0)
[2017-12-05 15:03:37] [memory] Extending reserved space to 2048 MB (device 1)
[2017-12-05 15:03:37] Training started
[2017-12-05 15:03:37] [memory] Reserving 490 MB, device 0
[2017-12-05 15:03:39] [memory] Reserving 490 MB, device 1
...
[2017-12-05 15:09:59] Ep. 1 : Up. 1000 : Sen. 81308 : Cost 84.54 : Time 382.09s : 2688.68 words/s
[2017-12-05 15:09:59] Saving model to model.iter1000.npz
[2017-12-05 15:10:05] Saving model to model.npz
...
```

Note: Doom and Adan machines are not binary compatible. If you get an error "Illegal instruction" when running Marian it means, that you compiled Marian on one machine, but tried running it on the other.

## Non-Interactive Training
Always use non-interactive jobs for long-time training. The main reason is that if your job dies, it frees the GPU. An interactive job would wait for you to continue.

Here are MetaCentrum instructions on submitting jobs. Remember you need to ask for GPUs (and doom machines).

Also remember that you need to `module add cuda-8.0` in the job script.

Translate with the First Saved Model
The training will take very long time. We can test any saved model, independently.

## Translating on GPU (marian)

In the following commands, you will have to use the corresponding vocabulary files that Marian created for you. Do not blindly copy-paste train.src.yml train.tgt.yml:

```
# interactively ask for 1 GPU for 1 hour
qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=1gb:cl_doom=True -l walltime=1:00:00 -I

marian-decoder   \
  --models model.iter1000.npz model.iter2000.npz    \
  --vocabs train.src.yml train.tgt.yml \
  < dev.src \
  > dev.translated-with-model-iter1000
```

For a huge speedup, use batched translation. Specifically, I saw these times on the test set of 2500 czengali sentences on doom machines (4.7GB GPU size):

```
No special batching flags                                      	Total time: 321.19093s wall
--mini-batch 64 --maxi-batch-sort src --maxi-batch 100 -w 2500 	Total time: 141.95220s wall
--mini-batch 64 --maxi-batch-sort src --maxi-batch 100 -w 4200 	Total time: 142.01441s wall
--mini-batch 300 --maxi-batch-sort src --maxi-batch 100 -w 4200	Total time: 243.57941s wall  # counter-intuitive: larger batch size takes more time
--mini-batch 256 --maxi-batch-sort src --maxi-batch 100 -w 4200	Total time: 219.79195s wall  # counter-intuitive: larger batch size takes more time
```

You can list several model files (`--model model.iter3000.npz model.iter4000.npz`), marian-decoder will ensemble them. You can even use different model types (transformer vs. s2s vs. amun).

## Translating on CPU (amun)

If you used the amun model type, you can use amun to translate with it.

```
# check if the machine is free enough with 'top'
# or ask for e.g. 4 CPUs with qsub
amun \
  --model model.iter1000.npz \
  --input-file dev.src \
  --source-vocab train.src.yml \
  --target-vocab train.tgt.yml \
  --cpu-threads 4 \
> dev.translated-with-model-iter1000
```
You can list several model files (`--model model.iter3000.npz model.iter4000.npz`), amun will ensemble them.

## Post-processing the Translation
If you used subwords (BPE), i.e. for dataset B, you definitely need to remove them. Simply pass the output through sed 's/@@ //g'.

For final (human) evaluation, it would be also very important use cased outputs (not lowercase as dataset A) and to detokenize the MT outputs.

```
## THIS IS OPTIONAL, WE DO NOT NEED TO EVALUATE MANUALLY
# Download the detokenizer:
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl
chmod +x detokenizer.perl

# Join BPE and detokenize MT output
# The parameter "-u" ensures that sentences get capitalized
cat dev.translated-with-model-iter1000 \
| sed 's/@@ //g'  \
| detokenizer.perl -u \
> dev.translated-with-model-iter1000.detokenized.txt
```
