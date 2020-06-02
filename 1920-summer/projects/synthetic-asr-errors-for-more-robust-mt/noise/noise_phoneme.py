import argparse
import os
import subprocess
import re
from apply import G2PModelTester
from noise.noise_base import NoiseGenerator


def runG2PCommand(name, g2p_command):
    # print("Applying {} model...".format(name))

    with open(os.devnull, "w") as devnull:
        proc = subprocess.Popen(
            g2p_command,
            stdout=subprocess.PIPE,
            stderr=devnull  # if not self.verbose else None
        )

        for line in proc.stdout:
            parts = re.split(r"\t", line.decode("utf8").strip())
            if not len(parts) == 3:
                parts = [parts[0], None, None]
            yield parts

    return


def runP2GCommand(name, g2p_command):
    with open(os.devnull, "w") as devnull:
        proc = subprocess.Popen(
            g2p_command,
            stdout=subprocess.PIPE,
            stderr=devnull  # if not self.verbose else None
        )

        for line in proc.stdout:
            parts = re.split(r"\t", line.decode("utf8").strip())
            if not len(parts) == 3:
                parts = [parts[0], None, None]
            yield parts

    return


def generate_splits(phonemes_list, cnt_parts):
    assert cnt_parts > 1, "number of the parts have to be at least 2"

    all_splits = []
    phonemes_list_vect = phonemes_list.split()
    if len(phonemes_list) < cnt_parts:
        return all_splits  # not possible to split X phonemes to Y parts if X < Y

    def splitter(str):
        for i in range(1, len(str)):
            start = str[0:i]
            end = str[i:]
            yield (start, end)
            for split in splitter(end):
                result = [start]
                result.extend(split)
                yield result

    combinations = list(splitter(phonemes_list_vect))

    # fix [['HE'], ['LL', 'OW']]  --> ['HE', 'LO WW']
    for split in combinations:
        split = list(split)
        final_split = []
        for part in split:
            s = " ".join(part)
            final_split.append(s)
        if len(final_split) == cnt_parts:
            all_splits.append(final_split)

    return all_splits  # [  ['HE', 'LO WW'] , ['HE LO', 'WW']  ]

class NoiseFromP2G(NoiseGenerator):
    def __init__(self, g2p_model_path, p2g_model_path, pronounc_dict_path, cnt_error_samples, max_N):
        NoiseGenerator.__init__(self)
        self.max_N = max_N
        self.cnt_error_samples = cnt_error_samples

        self.g2p_model_path = g2p_model_path
        self.p2g_model_path = p2g_model_path
        self.pronounc_dict_path = pronounc_dict_path

        self.temp_g2p_source = "tmp.words.txt"
        self.temp_p2g_source = "tmp.pronounc.txt"

    def get_name(self):
        return "phoneme"

    # word -> pronunciation
    def generate_pronunciation(self, words_alpha_pos):
        with open(self.temp_g2p_source, "w") as file:
            for w in words_alpha_pos:
                word, _, id = w
                file.write(word.lower())
                file.write('\n')

        return self.run_g2p_inference()

    # pronunciation -> word
    def generate_noise(self, list_sampling_nodes):
        """
        generate P2G using the phonetisaurus script
        parses the output
        :param list_sampling_nodes:
        :return: results: { node_id : {to_n : [(error, score)] } }
        """
        with open(self.temp_p2g_source, "w") as file:
            for node in list_sampling_nodes:
                for j in range(self.max_N):
                    N = j + 1

                    # M:1 error
                    if N == 1:
                        file.write("\\--- {}\n".format(node.sampling_id))  # ----- 134 1
                        file.write(node.phonemes)
                        file.write('\n')

                    # M:N error
                    else:
                        all_splits = generate_splits(node.phonemes, N)
                        # [  ['HE', 'LO WW'] , ['HE LO', 'WW']  ]
                        for split_id, split in enumerate(all_splits):
                            for split_part_id, split_part in enumerate(split):
                                file.write("\\--- {} {} {} {}\n".format(node.sampling_id, N, split_part_id + 1, split_id))  # ----- 17 3 1 0  # node_id , N, n..N, variant of n (e.g. AH OJ, AHO J)
                                file.write(split_part)
                                file.write('\n')

        # now run the inference
        raw_results = self.run_p2g_inference()
        parsed_results = self.parse_p2g_results(raw_results)
        return parsed_results

    def parse_p2g_results(self, results):
        # { node_id : {to_n : [] } }
        self.parsed_results = {}
        node_id = None
        to_N = -1
        part_n = -2
        variant_n = -3
        ready_to_dump = False
        to_N_part_XX_variant_v = {}

        for result in results:
            phonemic_word, score, synthetised_word = result

            # id sequence
            if phonemic_word.startswith("\\---"):
                ready_to_dump, node_id, to_N, part_n, variant_n, to_N_part_XX_variant_v, phonemic_word = self.parse_id_sequence(
                    ready_to_dump, node_id, to_N, part_n, variant_n, to_N_part_XX_variant_v, phonemic_word)

            elif score is None:
                assert score is None and synthetised_word is None, "Score none but synth word exists"
                continue

            else:
                if to_N == 1:
                    # self.map_id_sampling_node[node_id].add_sample(to_N, synthetised_word, score)
                    self.add_parsed_result(node_id, to_N, synthetised_word, score)

                else:
                    if to_N == part_n:
                        ready_to_dump = True  # we are theoretically ready to dump ...
                    if part_n in to_N_part_XX_variant_v:
                        to_N_part_XX_variant_v[part_n].append((synthetised_word, score))  # = {"part_n": []}
                    else:
                        to_N_part_XX_variant_v[part_n] = [(synthetised_word, score)]
        return self.parsed_results

    def add_parsed_result(self, node_id, to_N, synthetised_word, score):
        if node_id in self.parsed_results:
            if to_N in self.parsed_results[node_id]:
                self.parsed_results[node_id][to_N].append((synthetised_word, score))
            else:
                self.parsed_results[node_id][to_N] = [(synthetised_word, score)]
        else:
            self.parsed_results[node_id] = {to_N: [(synthetised_word, score)]}

    def parse_id_sequence(self, ready_to_dump, node_id, to_N, part_n, variant_n, to_N_part_XX_variant_v, phonemic_word):
        if ready_to_dump:
            self.add_parsed_MN_error(node_id, to_N, part_n, variant_n, to_N_part_XX_variant_v)
            to_N_part_XX_variant_v = {}
            to_N = -1
            part_n = -2
            variant_n = -3
            ready_to_dump = False

        parts = phonemic_word.split()
        assert len(parts) == 2 or len(parts) == 5, "Result of p2g is wrong..."

        if len(parts) == 2:  # M : 1
            to_N = 1
            node_id = int(parts[1])

        else:  # M : N
            node_id = int(parts[1])
            to_N = int(parts[2])
            part_n = int(parts[3])
            # print("setting part n {}".format(part_n))
            variant_n = int(parts[4])

        return ready_to_dump, node_id, to_N, part_n, variant_n, to_N_part_XX_variant_v, phonemic_word

    def add_parsed_MN_error(self, node_id, to_N, part_n, variant_n, to_N_part_XX_variant_v):
        to_N = int(to_N)
        # print("{} {} {} {}".format(node_id, to_N, part_n, variant_n))
        assert part_n == to_N, "Safety check, n and N must be equal in order to dump ...."

        options = []

        def add_to_option(options, i, sample):
            wr_old, pr_old = options[i]
            wr, pr = sample
            pr = float(pr)
            options[i] = (wr_old + " " + wr, pr_old + pr)

        for key, value in to_N_part_XX_variant_v.items():
            part_n_of_N = key  # not needed ...
            list_of_samples = value

            if len(options) == 0:
                for i in range(len(list_of_samples)):
                    w, p = list_of_samples[i]
                    p = float(p)
                    options.append((w, p))
            else:
                if len(options) > len(list_of_samples):
                    for i in range(len(options)):
                        if i < len(list_of_samples):
                            add_to_option(options, i, list_of_samples[i])
                        else:
                            add_to_option(options, i, list_of_samples[-1])
                elif len(options) < len(list_of_samples):
                    for i in range(len(options)):
                        add_to_option(options, i, list_of_samples[i])
                else:
                    for i in range(len(list_of_samples)):
                        add_to_option(options, i, list_of_samples[i])

        for option in options:
            synthetised_word, score = option
            self.add_parsed_result(node_id,to_N, synthetised_word, score / to_N)
            # self.map_id_sampling_node[node_id].add_sample(to_N, synthetised_word, score / to_N)

    def run_g2p_inference(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--model", "-m", help="Phonetisaurus G2P fst model.", default=self.g2p_model_path)
        parser.add_argument("--lexicon", "-l", help="Optional reference lexicon.", default=self.pronounc_dict_path)
        parser.add_argument("--nbest", "-n",
                            help="Maximum number of hypotheses to produce.  Overridden if --pmass is set.", default=1,
                            type=int)
        parser.add_argument("--beam", "-b", help="Search 'beam'.", default=10000, type=int)
        parser.add_argument("--thresh", "-t", help="Pruning threshold for n-best.", default=99.0, type=float)
        parser.add_argument("--greedy", "-g", help="Use the G2P even if a reference lexicon has been provided.",
                            default=True, action="store_true")
        parser.add_argument("--accumulate", "-a", help="Accumulate probabilities across unique pronunciations.",
                            default=False, action="store_true")
        parser.add_argument("--pmass", "-p",
                            help="Select the maximum number of hypotheses summing to P total mass for a word.",
                            default=0.0, type=float)
        parser.add_argument("--probs", "-pr", help="Print exp(-val) instead of default -log values.", default=False,
                            action="store_true")
        parser.add_argument("--verbose", "-v", help="Verbose mode.", default=False, action="store_true")
        parser.add_argument("--gsep", help="separator of 'graphemes', default: ' '", default="")

        args = parser.parse_args()
        tester = G2PModelTester(args.model,
                                **{key: val for key, val in args.__dict__.items() if not key in ["model", "word_list"]})
        results = runG2PCommand("G2P", g2p_command=tester.makeG2PCommand(self.temp_g2p_source))

        return results

    def run_p2g_inference(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("--model", "-m", help="Phonetisaurus G2P fst model.", default=self.p2g_model_path)
        parser.add_argument("--lexicon", "-l", help="Optional reference lexicon.", required=False)
        parser.add_argument("--nbest", "-n",
                            help="Maximum number of hypotheses to produce.  Overridden if --pmass is set.",
                            default=self.cnt_error_samples, type=int)
        parser.add_argument("--beam", "-b", help="Search 'beam'.", default=10000, type=int)
        parser.add_argument("--thresh", "-t", help="Pruning threshold for n-best.", default=99.0, type=float)
        parser.add_argument("--greedy", "-g", help="Use the G2P even if a reference lexicon has been provided.",
                            default=False, action="store_true")
        parser.add_argument("--accumulate", "-a", help="Accumulate probabilities across unique pronunciations.",
                            default=False, action="store_true")
        parser.add_argument("--pmass", "-p",
                            help="Select the maximum number of hypotheses summing to P total mass for a word.",
                            default=0.0, type=float)
        parser.add_argument("--probs", "-pr", help="Print exp(-val) instead of default -log values.", default=False,
                            action="store_true")
        parser.add_argument("--verbose", "-v", help="Verbose mode.", default=False, action="store_true")
        parser.add_argument("--gsep", help="separator of 'graphemes', default: ' '", default=" ")

        args = parser.parse_args()
        tester = G2PModelTester(args.model,
                                **{key: val for key, val in args.__dict__.items() if not key in ["model", "word_list"]})
        results = runP2GCommand("P2G", g2p_command=tester.makeG2PCommand(self.temp_p2g_source))

        return results
