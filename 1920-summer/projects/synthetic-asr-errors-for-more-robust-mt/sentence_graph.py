import logging
import math
import random
import re

import nltk
from nltk.tokenize import word_tokenize

import logger
from noise.noise_phoneme import NoiseFromP2G
from utils import choice


class SamplingGraph:
    def __init__(self, noise_generator, error_prob, max_M, sampling_M, sampling_N, sampling_error_samples):
        self.logger = logging.getLogger(logger.BasicLogger.logger_name)
        self.noise_generator = noise_generator
        try:
            nltk.data.find('tokenizers/punkt')
        except:
            self.logger.warning('downloading nltk tokenizer punkt...')
            nltk.download('punkt')

        assert 0 < error_prob < 1, "error prob must be in (0,1)"
        assert max_M >= 1, "1:N errors are smallest possible option"
        assert sampling_M in ['weighted', 'uniform'], "Sampling M must be 'weighted' or 'uniform'"
        assert sampling_N in ['weighted', 'uniform'], "Sampling N must be 'weighted' or 'uniform'"

        # Parameters for the sampling of the errors
        self.max_M = max_M
        self.sampling_M = sampling_M
        self.sampling_N = sampling_N
        self.error_prob = error_prob
        self.sampling_error_samples = sampling_error_samples

    def tokenize_sentence(self, sentence):
        assert isinstance(sentence, str), "sentence must be non-empty string"
        assert len(sentence) > 0, "sentence must be non-empty string"
        # replace typographic stuff
        sentence = sentence.replace("’", "'")
        sentence = sentence.replace('“', '"')
        sentence = sentence.replace('”', '"')
        sentence = sentence.replace('“', '"')

        # tokenize sentence and extract alpha words + their position in the sentence
        words = word_tokenize(sentence)
        self.original_words = words
        self.words_alpha_pos = list([])
        for i, word in enumerate(words):
            if word.isalpha():
                self.words_alpha_pos.append((word.lower(), "", i))

        # if we use phoneme model, we must generate pronunciation for alpha words
        if self.noise_generator.get_name() == 'phoneme':
            pronunciation = self.noise_generator.generate_pronunciation(self.words_alpha_pos)
            for i, result in enumerate(pronunciation):
                w, _, idx = self.words_alpha_pos[i]
                self.words_alpha_pos[i] = (w, result[2], idx)

    def build_base_graph(self):
        self.end_node = Node((".", ".", len(self.original_words)), len(self.words_alpha_pos) + 1, from_M=-2)
        self.start_node = Node(("<eps>", "<eps>", -1), 0, from_M=-1)
        self.map_start_word_level_id__node = {}
        for i in range(0, len(self.words_alpha_pos) + 2):
            self.map_start_word_level_id__node[i] = []
        self.map_start_word_level_id__node[0].append(self.start_node)
        self.map_start_word_level_id__node[self.end_node.level_id].append(self.end_node)

    def set_sentence(self, sentence):
        """
        Weighted Oriented Acyclic graph is build from the sentence
        Each node represent one error instance
        All positions of the errors are considered using window=1
        Nodes of errors are connected s.t. only correct sentences are constructed
        Weights between noedes represents probability of the erro
        :param sentence:
        :return:
        """

        # 1. TOKENIZE SENTENCE
        self.tokenize_sentence(sentence)

        # 2. BUILD "CORRECT SENTENCE" GRAPH
        # sentence: Hello, how are you
        # graph:    <eps>:0 -->  HELLO:1 --> HOW:2 --> ARE:3, YOU:4 --> '.':5
        self.build_base_graph()

        # 3. BUILD REST OF THE GRAPH
        self.__build_noise_graph(self.words_alpha_pos)

        # 4. CREATE ERROR SAMPLES
        # { node_id : {to_n : [(error, score)] } }
        error_samples = self.noise_generator.generate_noise(self.list_sampling_nodes)
        for node_id, errors in error_samples.items():
            for to_N, err_score_list in errors.items():
                for err_score in err_score_list:
                    error_word, score = err_score
                    self.map_id_sampling_node[node_id].add_sample(to_N, error_word, score)

    def sample_sentence(self):
        """
        Example of how the final sentence is constructed:
                             0     1  2   3  4      5       6   7   8
        original sentence: Hello THIS ,   is 123   welcome ??? there !
                             0     1        3         5           7
        alpha sentence   : Hello THIS      is       welcome    there
                             0  1          3          5 7
        synth sentence   : [hellish ]   [is this] [welcomere]

                             0 1      2    3       4     5 7      8
        final sentecnce  : [Hellish] , [is this] 123 [welcomere] !
        we can see, that non alpha token '???' at position 6 in the original sentence have to be omitted
        also the letter casing is forgotten
        and finally, spacing between alpha and non-alpha characters (like punctuation symbols) will also be different
        """

        current_node = self.start_node
        debug_sentence = []
        synth_sentence = []

        while True:
            if current_node.from_M_variant == -2:  # ... '.'
                # sentence.append(".")
                break
            if current_node.from_M_variant == -1:
                pass  # sentence.append("<eps> ")

            elif current_node.from_M_variant == 0:  # original word
                debug_sentence.append(current_node.word + " ")
            else:  # sampled word ...
                # sentence.append(">" + current_node.word + "<[{}] ".format(current_node.variant))
                # we are sampling now ....
                to_N_parts = current_node.select_N(self.sampling_N)
                sampled_word = current_node.sample_word(to_N_parts, self.sampling_error_samples)
                synth_sentence.append((sampled_word, current_node.sentence_start_idx, current_node.from_M_variant))
                debug_sentence.append(">" + sampled_word + "<[{}->{}] ".format(current_node.from_M_variant,
                                                                               to_N_parts))  # current_node.variant

            # current_node.display()

            if len(current_node.neighbours) == 1:
                current_node = current_node.neighbours[0]
            else:
                current_node = choice(current_node.neighbours, current_node.weights)

        final_sentence = []
        last_idx_in = -1
        for synth_word in synth_sentence:
            word, start_idx, cnt_words = synth_word
            word = word.replace(".", "")
            if start_idx == 0:
                word = word.capitalize()

            # fill in missing  non-alpha words
            while 1 + last_idx_in < start_idx:
                last_idx_in += 1
                final_sentence.append(self.original_words[last_idx_in])

            # we fill current word
            assert start_idx == 1 + last_idx_in, "safety check"

            final_sentence.append(word)
            # wrong... last_idx_in = last_idx_in + cnt_words  # from 1 , from 2 ...
            # because we can occasionally skip  " this && is " => "thisis" ] and we need to check this ... here
            # ( we actually moved 3 words forward, not 2  )
            # we check it against the index in the alpha sentence
            for final_idx, w_p_idx in enumerate(self.words_alpha_pos):
                _, _, orig_idx = w_p_idx
                if orig_idx == start_idx:
                    _, _, true_idx = self.words_alpha_pos[final_idx + cnt_words - 1]
                    last_idx_in = true_idx
                    # synth word:      welcomere(5)
                    # words alpha pos: welcome(5)    there(7)
                    # last indx in : not 6
                    # but ... ....       7
                    break

        # fill the end of the sentence
        while last_idx_in + 1 < len(self.original_words):
            last_idx_in += 1
            final_sentence.append(self.original_words[last_idx_in])

        final_sentence = " ".join(final_sentence)
        # fix [ ( {
        final_sentence = re.sub(r'\s*([(\[{])\s*', r' \1', final_sentence)  # "is ( maybe ) good" -> "is (maybe ) good"
        # fix } ) }
        final_sentence = re.sub(r'\s*([)\]\}])\s*', r'\1 ', final_sentence)  # "is (maybe ) good" -> "is (maybe) good"
        # fix -  @
        final_sentence = re.sub(r'\s*([@])\s*', r'\1', final_sentence)  # "hello - kitty" -> "hello-kitty"
        # fix , . ; : ! ? % $
        final_sentence = re.sub(r'\s([?,.;!:%$](?:\s|$))', r'\1', final_sentence)  # " hello , Peter" -> hello, Peter

        # fix ``
        # final_sentence = re.sub(r'\s*(``)\s*', r' \1', final_sentence)
        final_sentence = re.sub(r"(``)\s", r'"', final_sentence)
        # fix ''
        final_sentence = re.sub(r"\s(''(?:\s|$))", r'" ', final_sentence)

        final_sentence = final_sentence.strip()
        # def remove_s(final_sentence, sym):
        #     p1 = -1
        #     for i, s in enumerate(final_sentence):
        #         if s == sym:
        #             if p1 == -1:
        #                 if i + 1 < len(final_sentence):
        #                     if final_sentence[i + 1] == " ":
        #                         if i == 0 or final_sentence[i - 1] == " ":
        #                             p1 = i
        #                             continue
        #             if p1 != -1:
        #                 if final_sentence[i-1] == " " and i-2 != p1:
        #                     final_sentence = final_sentence[0:p1+1] + final_sentence[p1+2: i-1] + final_sentence[i:]
        #                 else:
        #                     final_sentence = final_sentence[0:p1 + 1] + final_sentence[p1 + 2: ]
        #                 break
        #     return final_sentence
        #
        # cnt = final_sentence.count('"')
        # for i in range(math.floor(cnt/2)):
        #     final_sentence = remove_s(final_sentence, '"')

        # 012345678
        # A " B " C
        #   2   6
        # [0:3]
        debug_sentence = "\t".join(debug_sentence)
        return debug_sentence, final_sentence

    def __create_arc_weight(self, node, level):
        if node.from_M_variant == -1:  # end node...
            return 1
        elif node.from_M_variant == 0:  # 'original word'
            return 1 - self.error_prob

        elif len(self.map_start_word_level_id__node[level]) == 1:
            return 1

        else:
            cnt_on_level = len(self.map_start_word_level_id__node[level])

            if self.sampling_M == 'uniform':
                normed_weight = 1
            else:
                # 1:N -- weight = max_M - 1 + 1  = 3
                # 2:N -- weight = max_M - 2 + 1  = 2
                # 3:N  -- weight = max_M - 2 + 1 = 1
                weight = self.max_M - node.from_M_variant + 1
                sum = self.max_M * (self.max_M + 1) / 2
                normed_weight = weight / sum

            # error_prob is distributed between different types of errors
            prob = self.error_prob / (cnt_on_level - 1) * normed_weight  # -1 for the original word ...
            return prob

    def __build_noise_graph(self, words):
        """
        nodes for errors are build
        1:1 ... 1:N
        2:1 ... 2:N
        ... ... ...
        M:1 ... M:N
        Edges are build s.t. it creates correct sentence
        :param words:
        :return:
        """

        self.map_id_sampling_node = {}
        self.max_id = 0
        self.list_sampling_nodes = []

        for i in range(len(words) - 1, -1, -1):
            # because nodes ends with the same word, they all have same successors
            # on the other hand, they differ by the "start" level

            # A. 1:1 error
            list_new_nodes = [Node(words[i], level_id=i + 1, from_M=0)]

            # B. All other word 1:x 2:x ...
            for mapping in range(self.max_M):
                # mapping is typically 1:x, 2:x
                if i - mapping < 0:  # prevent underflow
                    break
                tmp_word = ""  # 0 .... words[i - 0]  , 1 ... words[i-1] words[i - 0]
                tmp_word_phonemes = ""
                first_idx = None

                for j in range(mapping, -1, -1):
                    if j < mapping:
                        tmp_word += " "
                        tmp_word_phonemes += " "
                    w, p, sentence_idx = words[i - j]
                    if first_idx is None:
                        first_idx = sentence_idx
                    tmp_word += w
                    tmp_word_phonemes += p

                word = (tmp_word, tmp_word_phonemes, first_idx)
                _new_node = Node(word, level_id=i + 1 - mapping, from_M=1 + mapping, sampling_id=self.max_id)
                list_new_nodes.append(_new_node)

                self.map_id_sampling_node[self.max_id] = _new_node
                self.list_sampling_nodes.append(_new_node)
                self.max_id += 1

            # C. Add all successors to all new nodes
            for successor in self.map_start_word_level_id__node[i + 2]:
                for new_node in list_new_nodes:
                    weight = self.__create_arc_weight(successor, level=i + 2)
                    new_node.add_neighbour(successor, weight)  # new_node ----weight---> successor

            # D. Add all new nodes to the map
            for new_node in list_new_nodes:
                self.map_start_word_level_id__node[new_node.level_id].append(new_node)

        # finally, when all nodes are set
        # add arcs from <eps> to all nodes representing start of the sentence
        for successor in self.map_start_word_level_id__node[1]:
            weight = self.__create_arc_weight(successor, level=1)
            self.start_node.add_neighbour(successor, weight)

class Node:
    def __init__(self, word, level_id, from_M, sampling_id=0):  # word(s), already as the list of phonemes
        # word stuff
        self.from_M_variant = from_M  # -2=end, -1=start, 0=original ; 1=1:N ; 2=2:N 3=3:N ...
        self.word, self.phonemes, self.sentence_start_idx = word
        self.target_samples = {}  # M to 1: 2: 3:  #  {1: [("heck", 0.221), ("hrrck", 0.4), ("hrrrr", 0.17)], 2: [("he hr", 0.01)], 3: []}
        self.cnt_samples = 0  # we start with zero samples ....
        self.sampling_id = sampling_id
        # node stuff
        self.level_id = level_id
        self.neighbours = []
        self.weights = []
        self.set_word = set(self.word.split())
        self.logger = logging.getLogger(logger.BasicLogger.logger_name)

    def select_N(self, sampling_N):
        # select N
        # we do it weighted.... the higher the N, the lower the probability

        if len(self.target_samples) == 0:
            # self.logger.warning("for id:{} word:'{}' empty targets, returning -1".format(self.sampling_id, self.word))
            return -1

        N_list = list(self.target_samples.keys())  # 1 2 4  ... 7

        if sampling_N == 'uniform':
            i = random.randint(0, len(N_list) - 1)
            N = N_list[i]
        else:
            _sum = sum(N_list)
            weights = [_sum - v + 1 for v in N_list]
            N = choice(N_list, weights)

        return N

    def sample_word(self, to_N, sampling_error_samples):
        # empty targets...
        if to_N == -1:
            return self.word

        # print("word: {}".format(self.word))
        if to_N in self.target_samples:

            possibilities = self.target_samples[to_N]

            if sampling_error_samples == 'uniform':
                i = random.randint(0, len(possibilities) - 1)
                return possibilities[i]

            # sample word by probability ....
            wrds = []
            weights = []

            for poss in possibilities:
                w, p = poss
                wrds.append(w)
                weights.append(1 / p)

                # if self.word not in w.split():
                #     narrower.append((w, p))
                # print(poss)
            weights = [float(i) / max(weights) for i in weights]

            selected = choice(wrds, weights)
            return selected

        # this won't happen is N i selected by Node function select_N
        else:
            self.logger.warning(
                "for id:{} word:'{}' target_meta to_N[{}] not available".format(self.sampling_id, self.word, to_N))
            self.logger.warning("{}".format(self.target_samples))

            # Just select randomly some other [available] key...
            N_list = list(self.target_samples.keys())
            i = random.randint(0, len(N_list) - 1)
            N = N_list[i]
            return self.sample_word(N, sampling_error_samples)

    def add_sample(self, to_N, synthetised_word, probability):
        probability = float(probability)
        # prevent same
        if synthetised_word == self.word:
            return

        synth_set = set(synthetised_word.split())
        if len(self.set_word.intersection(synth_set)):  # maybe too restricting ... ?
            return
        # print("adding synthetised word to id: {}, word: '{}' , sampled_word: {}".format(self.sampling_id, self.word, synthetised_word))
        # add synthetised word
        if to_N in self.target_samples:
            self.target_samples[to_N].append((synthetised_word, probability))
        else:
            self.target_samples[to_N] = [(synthetised_word, probability)]

        # one more sample
        self.cnt_samples += 1

    def add_neighbour(self, node, weight):
        self.neighbours.append(node)
        self.weights.append(weight)

    def display(self):
        print(self.word)
        for k in range(len(self.neighbours)):
            neig = self.neighbours[k]
            neig_val = self.weights[k]
            print("\t p:{:.2f}, w: {}".format(neig_val, neig.word))


