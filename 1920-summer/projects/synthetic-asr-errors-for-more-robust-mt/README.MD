## Homo-Noiser ##

This repository contains scripts suitable for simulating ASR errors.


#### Contact: ####
  * martin.jenc@gmail.com
  * sunitbhattacharya.official@gmail.com 
 
 
#### Inatallation ####

First, install following python packages
```shell script
$ pip3 install jellyfish jiwer nltk numpy scipy tensorflow torch 
```

Next, you need to compile and install `mitlm` `openfst` and my dirty modified version of the Phonetisaurus. 
The source code for modified Phonetisaurus is in `_extra_src/modified-phonetisaurus`.
There is shell script which should do the installation. 
Read the `tran_phoneme/README.MD` where is detailed description of what to do.
 

### Data Preparation ###
In order to produce noise, we need to prepare some Noise models. 
Homo-Noiser will support three models in the final version, currenty two of them are fully implemented.
Homo-Noiser works with one model at a time, so you do not need to prepare all three at once.

#### Preprare Phoneme Models ####
Instructions are in `tran_phoneme/README.MD`. If you have successfully installed Phonetisaurus G2P the training of the model is fast (few minutes) and straightforward and there are shell scripts that will do it

```shell script
$ cd train_phoneme
$ ./download_and_format_cmudict.sh
$ ./train_g2p.sh
$ ./train_p2g.sh
```
Results of the scripts will be 
```
./train_phoneme/cmudict.formatted.dict ... lexicon
./train_phoneme/model/g2p.fst ... g2p model
./train_phoneme/model/p2g.fst ... p2g model
```

These three files will be passed as arguments to the Homo-Noiser script

#### Preprare Dictionary Models ####
Instructions are in `tran_dictionary/README.MD`. Few small dictionaries are already present. Aligning large dictionaries can be very time consuming (multiple days),
so instead of running the scripts and aligning it by yourself, you can download aligned dictionaries from my google drive:

**hashed dictionaries**
```shell script
# Download hashed dictionaries
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt

# Unzip and move to correct location
unzip hashed.zip
mv *.pkl ./train_dictionary/hashtable/
```

**aligned dictionaries**
```shell script
# Download aligned dictionaries
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt

# Unzip and move to correct location
unzip hashed.zip
mv *.pkl ./train_dictionary/dictionary/
```


Finally you need to create file with list of the used dictionaries. This File will be passed as the argument to the Homo-Noiser script.
There is already default file - `./dictionary-files-list.txt` where you can see the proper syntax.
Edit the file s.t. it match provided dictionaries

##### Preprare Embedding Models #####
Not in this version of Homo-Noiser yet 

#### Language Model Preparation ####
We use pre-trained BERT available from torch repo. You do not need to do anything.

#### Inference ####
All parameters of Homo-Noiser have default values.
If you prepare correct Error Model files and input files, it should run out of the box.
Here is list of all available parameters:

**Input files & output files**

Set this to your source files. Current sentence generator expects 1 directory `input_source_dir` with multiple files in it. Names of the files must be in separate file. 
Name of this file is passed as `input_filename_list`
In one file, we expect to have be single sentence on a line.
Finally we need to provide target directory `base_target_dir`. In this directory the files with noise will be crated. Name of the files will match the source file names....

```
# Input files:
--input_source_dir    default="example-input"  help="Base directory with the source files. !! 1 line = 1 sentence !!"
--input_filename_list default="file-list.txt"  help="File with list of all source files"
# Target directory:
--base_target_dir    default="example-output" help="Directory for nosified files"
```

If you have different files (for instance for CzEng), write your own generator (possibly by modifying current `SentencesFromListOfFiles` generator) 


**Main error parameters**

max_m max_n can should be left on default values in most of the cases ... works great as is
error_rate  -- set this as you wish
error_mode -- 'phoneme' model is much slower than 'dictionary' but hopefully provide better results (some experiments with this would be great)
note that `embedding` model is not yet present in this version of Homo-Noiser
```
--max_m       default=2       help="M:N errors. maximum number of source words (M) in single error."
--max_n       default=2       help="M:N errors. maximum number of target_meta words (N) in single error."
--error_rate  default=0.3     help="Probability of error. Because we have M:N errors and not only 1:1 errors, WER may be a bit higher then error rate."
--error_model default=phoneme help="How error is generated; options:['phoneme', 'dictionary', 'embedding']. Currently only phoneme is implemented here..."
```

**Additional error parameters** 

Some other useful parameters.
If you use `error_model=dictionary` you can set `error_samples` quite high (we sometimes have over 1000 similar from which we can sample words ...)
The default behaviour prefer to do 1:1 errors and higher order errors are less probable.
Moreover for each error we sample some reasonable amount of variants and select the more probable options.
```
--sampling_m             default='weighted' help="How 'M' is sampled for M:N error; options:['uniform','weighted']; 'weighted' := higher the M, lower the probability)")
--sampling_n             default='weighted' help="How 'N' is sampled for M:N error; options:['uniform','weighted']; 'weighted' := higher the N, lower the probability)")
--error_samples          default=5          help="How many different errors are generated for singe error (we sample from this)... increases variance of error")
--sampling_error_samples default='weighted' help="How error is sampled from available error samples; options:['weighted', 'uniform']. 'weighted' := higher the error score (score depends on error model), higher the probability")

```

**LM model parameters**

LM should improve the error sentences so they look alright. But inference is 2x slower.
Some experiments if using LM really improves the errors of the sentences would be great ...
```
--use_lm  default=True                 help="Use the language model or not ..."
--bert_lm default="bert-base-uncased"  help="Which pre-trained Bert to choose from available torch models"
```

**Sentence level parameters**

Again, this can also be left as is. The goal is to sample some reasonable amount of noisified sentences for single original sentence, s.t. we can select some probable ones.

But if you won't use Language model  ( `--use_lm=False` ) then you can set `--sentence_samples=1` because we have no score for selecting from multiple samples.
```
--min_wer                   default=0.1 help="Limit the variance of the WER across the sentences"
--max_wer                   default=0.6 help="Limit the variance of the WER across the sentences."
--sentence_samples          default=20  help="How many different errors are generated for singe error (we sample from this)... increases variance of error"
--sampling_sentence_samples default="weighted_lm" help="How final sentence is sampled options:['uniform','weighted_lm', 'max_lm']; 'weighted_lm' := higher the LM score, higher the probability)")
```

**Phoneme model parameters**

Set parameters you need to set if `error_model=phoneme`

For `error_model=phoneme` you need to specify:
```
--g2p_model  default="train_phoneme/model/g2p.fst"           help="Path to the trained g2p model (word -> phonemes)"
--p2g_model  default="train_phoneme/model/p2g.fst"           help="Path to the trained p2g model (phonemes -> word)"
--lexicon"   default="train_phoneme/cmudict.formatted.dict"  help="Dictionary for generating pronunciation of known word"
```

**Dictionary model parameters**

For `error_model=dictionary` you need to specify:

```
--dictionary_filename_list default="dictionary-files-list.txt" help="Path to the file which contains list of files to be used by dictionary error_model"
--jaro_winkler_threshold   default=0.8 help=""Words from dictionaries are consider similar (suitable for error) only when Jaro-Winkler similaryty >= 0.8"."

```

Keep in mind that setting for example `jaro_winkler_threshold=0.5` will not have any effect on "mra aligned dictionaries" which were crated with jaro_winkler similaryty threshold >= 0.8")
If you have downloaded dictionaries, you can see that right now there is `words-400K_mra_0.8.pkl` which is dictionary aligned with threshold `0.8`.

Other types of dictionaries (hash dictionaries / scraped dictionaries ...) are not aligned using mra + threshold, so setting `jaro_winkler_threshold`
to any value may have effect on them -- so it make sense to set this value to any number (if at least one such dictionary is used).

**Embedding model parameters**

For `error_model=embedding` you need to specify:
```
# todo -- embedding model not yet here ...
```

#### Experiments ####
* I ran no experiments at all (beyond some micro debugging tests)
* In `run_statistics_on_results.py` is simple scripts that calcualtes WER over source and target files ...
* Any other experiments and  statistics are welcome ...

#### Things to consider ####
* On my machine, using Language Model results in really slow generation ...
* Same with phoneme model
* It would be great to do some experiments if it will run much faster on cluster
* And also, if it is even worth it (is the noise really that much better?)

#### TODO ####
* write input generator for CzEng 2.0 (I think simple modification of `SentencesFromListOfFiles` generator should do the trick )
* run some experiments as described above and select some parameters / models ...
* process CzEng 2.0
