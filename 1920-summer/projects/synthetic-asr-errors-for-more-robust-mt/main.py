import argparse
import ntpath
import os
import re
import sys
import time

from jiwer import wer

from generators.factory import GeneratorFactory as generator_factory
import logger
import utils
from language_model import LM
from noise.noise_dict import NoiseFromDict
from noise.noise_phoneme import NoiseFromP2G
from run_statistics_on_results import wer_over_files
from sentence_graph import SamplingGraph


class HomoNoiserScript:

    def __init__(self, generator, **kwargs):
        self.verbose = kwargs.get("verbose", False)

        # Error params
        self.max_M = kwargs.get("max_m", 2)
        self.max_N = kwargs.get("max_n", 2)
        self.sampling_m = kwargs.get("sampling_m", "weighted")
        self.sampling_n = kwargs.get("sampling_n", "weighted")
        self.error_rate = kwargs.get("error_rate", 0.3)
        self.error_model = kwargs.get("error_model", 'phoneme')
        self.cnt_error_samples = kwargs.get("error_samples", 5)
        self.sampling_error_samples = kwargs.get("sampling_error_samples", "weighted")

        # Sentence error parameters
        self.min_wer = kwargs.get("min_wer", 0.1)
        self.max_wer = kwargs.get("max_wer", 0.6)
        self.cnt_sentence_samples = kwargs.get("sentence_samples", 10)
        self.sampling_sentence_samples = kwargs.get("sampling_sentence_samples", "weighted_lm")
        self.use_lm = kwargs.get("use_lm", True)
        self.lm_name = kwargs.get("bert_lm", None)

        # Phoneme model parameters
        self.g2p_model = kwargs.get("g2p_model", None)
        self.p2g_model = kwargs.get("p2g_model", None)
        self.lexicon = kwargs.get("lexicon", None)

        # Dictionary model parameters
        self.dictionary_filename_list = kwargs.get("dictionary_filename_list", None)
        self.jaro_winkler_threshold = kwargs.get("jaro_winkler_threshold", 0.8)
        # Embedding model parameters
        # ---

        # Target parameters
        self.base_target_dir = kwargs.get("base_target_dir", None)

        # Set logger
        self.logger = logger.BasicLogger.setupLogger(verbose=self.verbose)

        # Check everything is OK
        self.checkConfig()

        # Set Generator
        self.generator = generator

        noise_generator = self.get_noise_generator()

        self.sentence_graph = SamplingGraph(noise_generator=noise_generator,
                                            error_prob=self.error_rate,
                                            max_M=self.max_M,
                                            sampling_M=self.sampling_m,
                                            sampling_N=self.sampling_n,
                                            sampling_error_samples=self.sampling_error_samples)

        # Check if target directory is empty
        self.input_filename_list = kwargs.get("input_filename_list", None)
        self.input_source_dir = kwargs.get("input_source_dir", None)

        self.check_target_directory(self.base_target_dir, self.input_filename_list)

        # Set LM
        if self.use_lm:
            self.logger.info('loading {} model'.format(self.lm_name))
            self.bert_lm = LM(self.lm_name)
        else:
            self.bert_lm = None
            self.logger.info("Language model not used")

    def delete_files(self, list_of_filenames):
        import shutil
        for file_path in list_of_filenames:  # os.listdir(folder):
            # file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def check_target_directory(self, base_target_dir, file_list):
        """
        https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
        """

        with open(file_list) as f:
            files_list = [line.rstrip() for line in f]
        list_of_filenames = [os.path.join(base_target_dir, f) for f in files_list]

        existing_files = []
        for file_path in list_of_filenames:
            if os.path.isfile(file_path):
                existing_files.append(file_path)

        if len(existing_files) > 0:
            self.logger.error("!!!!! Target directory already contains target files !!!!! ----- Should the files be deleted ????? [Y/N]")
            time.sleep(0.5)
            print("delete files? [Y/N]: ", end="")
            while True:
                input1 = input()
                input1 = input1.lower()
                if input1 == "y":
                    self.logger.warning("Deleting files ...")
                    self.delete_files(existing_files)
                    break
                if input1 == "n":
                    self.logger.error("Move files somewhere else ... exiting now!")
                    exit(1)
                print("delete files? [Y/N]: ", end="")
                # self.logger.error("????? Should the files be deleted ????? [Y/N]")

    def get_noise_generator(self):
        if self.error_model == "phoneme":
            return NoiseFromP2G(g2p_model_path=self.g2p_model,
                                p2g_model_path=self.p2g_model,
                                pronounc_dict_path=self.lexicon,
                                cnt_error_samples=self.cnt_error_samples,
                                max_N=self.max_N)

        elif self.error_model == "dictionary":
            return NoiseFromDict(db_file_list=self.dictionary_filename_list,
                                 threshold=self.jaro_winkler_threshold,
                                 cnt_error_samples=self.cnt_error_samples,
                                 max_N=self.max_N)
        else:
            self.logger.error("Error model not implemented: {}".format(self.error_model))
            raise NotImplementedError

    def which(self, program):
        """Basic 'which' implementation for python.

        Basic 'which' implementation for python from stackoverflow:
          * https://stackoverflow.com/a/377028/6739158
        """

        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file

        return None

    def validateLexicon(self):
        validator_pattern = u"[\\}\\|_]"  # python2: unicode, python3: str
        validator = re.compile(validator_pattern)

        with open(self.lexicon, "r") as ifp:
            for line in ifp:
                if validator.search(line):
                    error = "Bad line contains reservered character:\n\t{0}"
                    error = error.format(line)
                    raise ValueError(error)

        return

    def checkConfig(self):
        self.logger.info("Checking command configuration...")
        for program in ["phonetisaurus-g2pfst", "phonetisaurus-align", "phonetisaurus-arpa2wfst"]:
            if not self.which(program):
                raise EnvironmentError(
                    ", ".join([
                        "Phonetisaurus command, '{0}'",
                        "not found in path."
                    ]).format(program)
                )

        # Create target_meta directory if not exists
        if not os.path.isdir(self.base_target_dir):
            self.logger.debug("Directory does not exist.  Trying to create.")
            os.makedirs(self.base_target_dir)

        if self.error_model == 'phoneme':
            self.logger.info("Checking lexicon for reserved characters: '}', '|', '_'...")
            self.validateLexicon()

        # Basic assertions
        if self.max_M < 1:
            self.logger.error("max_M must be >= 1, but {} given".format(self.max_M ))
            raise ValueError
        if self.max_N < 1:
            self.logger.error("max_N must be >= 1, but {} given".format(self.max_N))
            raise ValueError
        if self.cnt_error_samples < 1:
            self.logger.error("cnt_error_samples must be >= 1, but {} given".format(self.cnt_error_samples))
            raise ValueError
        if not 0.0 <= self.error_rate <= 1.0:
            self.logger.error("error_rate must be in [0,1], but: {} was given".format(self.error_rate))
            raise ValueError
        if not 0.0 <= self.min_wer <= 1.0:
            self.logger.error("min_wer must be in [0,1], but: {} was given".format(self.min_wer))
            raise ValueError
        if not 0.0 <= self.max_wer <= 1.0:
            self.logger.error("max_wer must be in [0,1], but: {} was given".format(self.max_wer))
            raise ValueError
        if self.cnt_sentence_samples < 1:
            self.logger.error("cnt_sentence_samples must be >= 1, but {} given".format(self.cnt_sentence_samples))
            raise ValueError

        # Other basic assertions
        if self.sampling_m not in ['uniform', 'weighted']:
            self.logger.error("sampling_m options are {}".format(['uniform', 'weighted']))
            raise ValueError
        if self.sampling_n not in ['uniform', 'weighted']:
            self.logger.error("sampling_n options are {}".format(['uniform', 'weighted']))
            raise ValueError
        if self.error_model not in ['phoneme', 'dictionary', 'embedding']:
            self.logger.error("sampling_n options are {}".format(['phoneme', 'dictionary', 'embedding']))
            raise ValueError
        if self.sampling_error_samples not in ['weighted', 'uniform']:
            self.logger.error("sampling_error_samples options are {}".format(['weighted', 'uniform']))
            raise ValueError
        if self.sampling_sentence_samples not in ['uniform', 'weighted_lm', 'max_lm']:
            self.logger.error("sampling_sentence_samples options are {}".format(['uniform', 'weighted_lm', 'max_lm']))
            raise ValueError

        if self.error_model == 'phoneme':
            if not os.path.isfile(self.p2g_model):
                self.logger.error("p2g_model not found: {}".format(self.p2g_model))
                raise FileNotFoundError
            if not os.path.isfile(self.g2p_model):
                self.logger.error("g2p_model not found: {}".format(self.g2p_model))
                raise FileNotFoundError
            if not os.path.isfile(self.lexicon):
                self.logger.error("lexicon not found: {}".format(self.lexicon))
                raise FileNotFoundError

        if self.error_model == 'dictionary':
            if not os.path.isfile(self.dictionary_filename_list):
                self.logger.error("dictionary_filename_list not found: {}".format(self.dictionary_filename_list))
                raise FileNotFoundError

        if self.error_model == 'embedding':
            self.logger.error("error model: {} not yet implemented".format(self.error_model))
            raise NotImplementedError

        items = vars(self).items()
        for key, val in sorted(items):
            self.logger.debug(u"{0}:  {1}".format(key, val))
        return

    def run(self):
        # For all sentences in the dataset ...
        for s in input_sentence_generator:
            source_doc_path, sentence_id, sentence = s

            # We must add sentence to the graph
            self.sentence_graph.set_sentence(sentence)

            # Score of the original sentence (only for debug purposes ...)
            if self.verbose:
                if self.bert_lm is not None:
                    score = self.bert_lm.get_score(sentence)
                    self.logger.debug("LM[{:.2f}]{}".format(score, sentence))
                else:
                    self.logger.debug("LM[ - ]{}".format(sentence))

            # Now we generate multiple (cnt_sentence_samples) "noisified variants" from current sentence
            avg_wer = 0
            samples_list = []

            # A. Crate samples
            tries = -1
            while True:
                tries += 1

                # all samples collected ( or we  do not want to wait too long ... )
                if len(samples_list) == self.cnt_sentence_samples or tries*2 > self.cnt_sentence_samples:
                    break

                debug, sample = self.sentence_graph.sample_sentence()
                if self.bert_lm is not None:
                    score = self.bert_lm.get_score(sample)
                else:
                    score = 1.
                error = wer(sentence, sample)

                if self.min_wer <= error <= self.max_wer :
                    avg_wer += error
                    samples_list.append((sample, score, error))

            # DEBUG: print the all sentence the variants
            if self.verbose:
                for sam in samples_list:
                    sample, score, error = sam
                    if self.bert_lm is not None:
                        self.logger.debug(" LM[{:.2f}] WER[{:.2f}]{}".format(score, error, sample))
                    else:
                        self.logger.debug(" LM[ - ] WER[{:.2f}]{}".format(error, sample))

                self.logger.debug("avg WER: {:.2f}".format(avg_wer / self.cnt_sentence_samples))

            # B. Finally we choose one sentence .....
            if len(samples_list) == 0:
                selected_sentence = sentence
            else:
                sentences = []
                lm_weights = []
                for s in samples_list:  # (sentence, LM, WER)
                    sentences.append(s[0])
                    lm_weights.append(s[1])
                selected_sentence = utils.choice(sentences, lm_weights)

            # And we write it to file
            source_doc = ntpath.basename(source_doc_path)
            target_doc_path = os.path.join(self.base_target_dir, source_doc)
            if os.path.isfile(target_doc_path):
                newline = True
            else:
                newline = False
            with open(target_doc_path, "a") as file:
                if newline:
                    file.write("\n")
                file.write(selected_sentence)
        self.logger.info("All files successfully processed")
        self.logger.info("Calculating WER on files...")
        time.sleep(0.5)
        return wer_over_files(self.input_source_dir, self.base_target_dir, self.input_filename_list)


def get_generator(**kwargs):
    input_source_dir = kwargs.get("input_source_dir", None)
    input_filename_list = kwargs.get("input_filename_list", None)

    # todo CzEng generator
    generator_class = "SentencesFromListOfFiles"
    generator = generator_factory.get_generator_class(generator_class, input_source_dir, input_filename_list)
    return generator


if __name__ == "__main__":
    """
    To run the HomoNoiser in with 'phoneme' error model you need to provide p2g, g2p models
    -> in train_phoneme are two scripts that generate it
    To run the HomoNoiser with 'dictionary' error model you need to provide dictionary (hash or direct)
    -> in train_dictionary are scripts that generate it
    """
    example = "{0} -er 0.3 -src example-input -lst input-files-list.txt -tg example-output --lexicon train_phoneme/cmudict.formatted.dict --p2g train_phoneme/model/p2g.fst  --g2p train_phoneme/model/g2p.fst ".format(
        sys.argv[0])
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--verbose", "-v", help="Verbose mode.", default=False, action="store_true")

    # Input files:
    parser.add_argument("--input_source_dir", "-src", type=str, default="example-input",
                        help="Base directory with the source files")
    parser.add_argument("--input_filename_list", "-lst", type=str, default="input-files-list.txt",
                        help="File with list of all source files")
    # Target directory:
    parser.add_argument("--base_target_dir", "-tg", type=str, default="example-output",
                        help="Directory for nosified files")

    # Main error parameters:
    parser.add_argument("--max_m", "-mm", type=int, default=2,
                        help="M:N errors. maximum number of source words (M) in single error.")
    parser.add_argument("--max_n", "-nm", type=int, default=2,
                        help="M:N errors. maximum number of target_meta words (N) in single error.")
    parser.add_argument("--error_rate", "-er", type=float, default=0.3,
                        help="Probability of error. Because we have M:N errors and not only 1:1 errors, WER may be a bit higher then error rate.")
    parser.add_argument("--error_model", "-em", type=str, default='phoneme',
                        help="How error is generated; options:['phoneme', 'dictionary', 'embedding']. Currently only phoneme is implemented here...")

    # Additional error parameters:
    parser.add_argument("--sampling_m", "-ms", type=str, default='weighted',
                        help="How 'M' is sampled for M:N error; options:['uniform','weighted']; 'weighted' := higher the M, lower the probability)")
    parser.add_argument("--sampling_n", "-ns", type=str, default='weighted',
                        help="How 'N' is sampled for M:N error; options:['uniform','weighted']; 'weighted' := higher the N, lower the probability)")
    parser.add_argument("--error_samples", "-ev", type=int, default=5,
                        help="How many different errors are generated for singe error (we sample from this)... increases variance of error")
    parser.add_argument("--sampling_error_samples", "-ses", type=str, default='weighted',
                        help="How error is sampled from available error samples; options:['weighted', 'uniform']. 'weighted' := higher the error score (score depends on error model), higher the probability")

    # LM model parameters
    parser.add_argument("--use_lm", "-ulm", type=bool, default=True, help="Use the language model or not ...")
    parser.add_argument("--bert_lm", "-lm", type=str, default="bert-base-uncased",
                        help="Which pre-trained Bert to choose from available torch models")

    # Sentence level parameters:
    parser.add_argument("--min_wer", "-miw", type=float, default=0.1,
                        help="Limit the variance of the WER across the sentences.")
    parser.add_argument("--max_wer", "-mw", type=float, default=0.6,
                        help="Limit the variance of the WER across the sentences.")
    parser.add_argument("--sentence_samples", "-ss", type=int, default=20,
                        help="How many different errors are generated for singe error (we sample from this)... increases variance of error")
    parser.add_argument("--sampling_sentence_samples", "-sss", type=str, default="weighted_lm",
                        help="How final sentence is sampled options:['uniform','weighted_lm', 'max_lm']; 'weighted_lm' := higher the LM score, higher the probability)")


    # Phoneme model parameters
    parser.add_argument("--g2p_model", "-g2p", type=str, default="train_phoneme/model/g2p.fst",
                        help="Path to the trained g2p model (word -> phonemes)")
    parser.add_argument("--p2g_model", "-p2g", type=str, default="train_phoneme/model/p2g.fst",
                        help="Path to the trained p2g model (phonemes -> word)")
    parser.add_argument("--lexicon", "-lex", type=str, default="train_phoneme/cmudict.formatted.dict",
                        help="Dictionary for generating pronunciation of known word")

    # Dictionary model parameters
    parser.add_argument("--dictionary_filename_list", "-dfl", type=str, default="dictionary-files-list.txt",
                        help="Path to the file which contains list of files to be used by dictionary error_model")
    parser.add_argument("--jaro_winkler_threshold", "-jwt", type=float, default=0.8,
                        help="Words from dictionaries are consider similar (suitable for error) only when Jaro-Winkler similaryty >= 0.8")

    # Embdedding model parameters

    args = parser.parse_args()

    input_sentence_generator = get_generator(**args.__dict__)
    homo_noiser_script = HomoNoiserScript(input_sentence_generator, **args.__dict__)
    homo_noiser_script.run()

