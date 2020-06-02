import logging
import os
import pickle
import jellyfish

# db_file = "honyme_db.pkl"
# noiser = HomophoneNoise(db_file)
import logger
from noise.noise_base import NoiseGenerator


class NoiseFromDict(NoiseGenerator):
    # database is created with words that have jaro_winkler threshold >= 0.5 ...
    def __init__(self, db_file_list, threshold=0.8, cnt_error_samples=5, max_N=1):  # >= 0.8
        self.logger = logging.getLogger(logger.BasicLogger.logger_name)

        NoiseGenerator.__init__(self)
        self.threshold = threshold
        if not 0.0 <= self.threshold <= 1.0:
            self.logger.error("jaro-winkler-threshold must be in [0,1], but: {} was given".format(self.threshold))
            raise ValueError

        self.cnt_error_samples = cnt_error_samples
        self.max_N = max_N

        # Parse file with list of the dictionaries
        hash_dicts, dicts = self.parse_file_list(db_file_list=db_file_list)

        # Load dictionaries to memory !!
        self.hash_dictionaries = []
        self.dictionaries = []

        self.load_dicits_to_memory(hash_dicts, dicts)

    def get_name(self):
        return "dictionary"

    def load_dicits_to_memory(self, hash_dicts, dicts):
        for hash_dict in hash_dicts:
            path, type_, from_M, to_N = hash_dict

            with open(path, 'rb') as handle:
                self.logger.info("Loading hashed dict: {}".format(path))
                _dict = pickle.load(handle)
                self.hash_dictionaries.append((_dict, type_, from_M, to_N))

        for dict in dicts:
            path, m, n = dict
            with open(path, 'rb') as handle:
                self.logger.info("Loading regular dict: {}".format(path))
                _dict = pickle.load(handle)
                self.dictionaries.append((_dict, m, n))

    def parse_file_list(self, db_file_list):

        hash_dicts = []
        dicts = []
        if not os.path.isfile(db_file_list):
            self.logger.error("Source list not found: {}".format(db_file_list))
            raise FileNotFoundError

        # load paths to dictionary files (and skip comments)
        with open(db_file_list) as f:
            files_list = [line.strip().split() for line in f]

        # remove comments and empty lines
        files_list[:] = [line for line in files_list if not (len(line) == 0 or line[0].startswith("#"))]


        for i, file_info in enumerate(files_list):
            if len(file_info) < 3 or len(file_info) > 4:
                self.logger.error("list with dictionary files: {} BAD SYNTAX ON LINE: {}".format(db_file_list, i + 1))
                raise SyntaxError

            path = file_info[0]
            if not os.path.isfile(path):
                self.logger.error("Path to dictionary file not found: {}".format(path))
                raise FileNotFoundError

            type_ = file_info[1].upper()
            if type_ not in ["DICTIONARY", "SOUNDEX", "METAPHONE", "MRA", "NYSIIS"]:
                self.logger.error(
                    "Unkown dictionary type: {} found in list with dictionary files: {}".format(type_, db_file_list))

            if type_ == "DICTIONARY":
                if len(file_info) != 4:
                    self.logger.error(
                        "list with dictionary files: {} BAD SYNTAX ON LINE: {}".format(db_file_list, i + 1))
                    raise SyntaxError

                from_M = float(file_info[2])
                to_N = float(file_info[3])
                dicts.append((path, from_M, to_N))

            else:
                if len(file_info) != 4:
                    self.logger.error(
                        "list with dictionary files: {} BAD SYNTAX ON LINE: {}".format(db_file_list, i + 1))
                    raise SyntaxError
                from_M = int(file_info[2])
                to_N = int(file_info[3])
                hash_dicts.append((path, type_, from_M, to_N))

        if len(hash_dicts) == 0 and len(dicts) == 0:
            self.logger.error("No dictionaries were found in dictionary list: {}".format(db_file_list))
            raise ValueError

        return hash_dicts, dicts

    def load_data(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def generate_pronunciation(self, words_alpha_pos):
        pass

    def recursive_merge(self, word_list, curr_depth, word_part, tmp):
        # word_list = []
        # curr_depth = 0
        # word_part  = (None,  0.0)
        # tmp =  [ [('Yolk', 1.0), ('and', 1.0)] ]
        if curr_depth == len(tmp):
            return [word_part]  # we must return list ...

        curr_level = tmp[curr_depth]  # [ (word, prob) ,  (word, prob) ... ]

        for i in range(len(curr_level)):
            wrd, score = word_part  # (None, 0.0)
            if wrd is None:
                wrd = ""
            else:
                wrd += " "
            wrd += curr_level[i][0]  # (word, prob)
            score += (curr_level[i][1] / len(tmp))  # average score
            new_word = (wrd, score)
            final_wrd = self.recursive_merge(word_list, curr_depth + 1, new_word, tmp)
            word_list.extend(final_wrd)  # extend otherwise ...

        return word_list

    def add_parsed_result(self, node_id, to_N, synthetised_word, score):
        if node_id in self.parsed_results:
            if to_N in self.parsed_results[node_id]:
                self.parsed_results[node_id][to_N].append((synthetised_word, score))
            else:
                self.parsed_results[node_id][to_N] = [(synthetised_word, score)]
        else:
            self.parsed_results[node_id] = {to_N: [(synthetised_word, score)]}

    def generate_noise(self, list_sampling_nodes):
        """
            generate P2G using the provided dictionary
            :param list_sampling_nodes:
            :return: results: { node_id : {to_n : [(error, score)] } }
        """
        self.parsed_results = {}

        for node in list_sampling_nodes:
            for j in range(self.max_N):
                N = j + 1
                M = node.from_M_variant
                list_words = []
                word = node.word
                # ......................................
                self.list_words_from_dicts(word, list_words, M=M, N=N)  # dict (M:N)
                self.list_words_from_hashes(word, list_words, N=N)     # hash(w1...wM) -> N (M:N)
                # ......................................
                noise_result = self.get_noise_results(orig_word=word, list_of_similar=list_words)

                if len(noise_result) == 0:
                    noise_result = self.fallback_M_all(word=word, M=M)
                    #if len(noise_result) == 0 and M > 1:
                    #    noise_result = self.fallback_split(word=word, N=N)
                # print("'{}':\t\t {}".format(word, noise_result))
                for result in noise_result:
                    synth_word, prob_score = result
                    self.add_parsed_result(node.sampling_id, N, synth_word, prob_score)

        # parsed results may be empty dictionary ...
        return self.parsed_results

    def get_hash(self, word, hash_type):
        word = word.replace(" ", "")
        if hash_type == "SOUNDEX":
            hash = jellyfish.soundex(word)
        elif hash_type == "NYSIIS":
            hash = jellyfish.nysiis(word)
        elif hash_type == "MRA":
            hash = jellyfish.match_rating_codex(word)
        elif hash_type == "METAPHONE":
            hash = jellyfish.metaphone(word)
        else:
            raise NotImplementedError("approach '{}' not implemented".format(hash_type))
        return hash

    def fallback_split(self, word, N):
        # print("fallback split")
        final_results = []
        split = word.split()
        if len(split) > N:
            split = split[:N]  # [w1,..wn]

        # add noise to individual words ... # w1: [w1_h1, w1_h2 ...]  ... wn : [wn_h1, wn_h2 ... ]
        tmp = []
        for w in split:
            ttmp = []
            self.list_words_from_dicts(w, ttmp, M=1, N=1)
            partial_result = self.get_noise_results(orig_word=w, list_of_similar=ttmp)
            tmp.append(partial_result)  # append not extend !!

        # and merge it back together ... all options [w1_h1 wm_h1 ...  ]  [w1_h1 wm_h2 ...  ] ....
        if len(tmp) > 0:
            results = self.recursive_merge([], 0, (None, 0.0), tmp)
            final_results.extend(results)
            # print("from split_merge [{}]:  {}".format(word, results))

        return final_results

    def fallback_M_all(self, word, M):
        # A. try try_similar_from_dicts M -> [..]
        lst_similar = []
        self.list_words_from_dicts(word, lst_similar, M=M, N=None)  # N=None ... to arbitrary N ...
        return self.get_noise_results(orig_word=word, list_of_similar=lst_similar)

    def list_words_from_dicts(self, word, lst_similar, M, N):
        for i, dic in enumerate(self.dictionaries):
            _dict, M__, N__ = dic
            if ( N is None and M == M__) or (M == M__ and N == N__):
                sim = self.similar_list_from_dict(word, i)
                if len(sim) > 0:
                    lst_similar.extend(sim)

    def similar_list_from_dict(self, word, dict_id):
        word = word.lower()
        if word in self.dictionaries[dict_id][0]:  # word exist # (dict,m,n)
            if len(self.dictionaries[dict_id][0]) > 0:  # safety
                return self.dictionaries[dict_id][0]
        return []

    def list_words_from_hashes(self, word, lst_similar, N):
        for i, tuple_ in enumerate(self.hash_dictionaries):
            if tuple_[3] == N:
                sim = self.similar_list_from_hash_dict(word, i)
                if len(sim) > 0:
                    lst_similar.extend(sim)

    def similar_list_from_hash_dict(self, word, hash_id):
        # _dict, type_, from_M, to_N
        hash_type = self.hash_dictionaries[hash_id][1]
        hashed = self.get_hash(word, hash_type)
        if hashed in self.hash_dictionaries[hash_id][0]:
            # for M:M we need at least 2 items in list ...
            if self.hash_dictionaries[hash_id][2] == self.hash_dictionaries[hash_id][3]:
                if len(self.hash_dictionaries[hash_id][0][hashed]) > 1:
                    return self.hash_dictionaries[hash_id][0][hashed]
            # for M:N 1 item is enough...
            else:
                if len(self.hash_dictionaries[hash_id][0][hashed]) >= 1:
                    return self.hash_dictionaries[hash_id][0][hashed]
        return []

    def get_noise_results(self, orig_word, list_of_similar):
        similar = []
        added_words = set({})
        for w2 in list_of_similar:
            if orig_word == w2:
                continue

            score = jellyfish.jaro_winkler_similarity(orig_word, w2)
            if score >= self.threshold and w2 not in added_words:  # and score must be higher than threshold
                similar.append((w2, score))  # word, pronunciation, score
                added_words.add(w2)

        # few or zero results ..
        if len(similar) < self.cnt_error_samples:
            return similar

        # else return top scoring
        # similar = sorted(similar, key=lambda entry: entry[1], reverse=True)  # by score from greatest
        return similar[:self.cnt_error_samples]
