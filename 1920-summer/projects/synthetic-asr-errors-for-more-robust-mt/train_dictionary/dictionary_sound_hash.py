import jellyfish
import ntpath

from train_dictionary.train_util import pickle_data, get_list_of_words


def get_hash(word, hash_type):
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


def create_hashed(word_list, from_M):
    all_dict = {}
    # FROM M
    if from_M == 1:
        for w in word_list:
            hash = get_hash(w, hash_to_use)
            all_dict[hash] = []

    elif from_M == 2:
        for w1 in word_list:
            for w2 in word_list:
                w = w1 + w2
                hash = get_hash(w, hash_to_use)
                all_dict[hash] = []
    else:
        raise NotImplementedError("from M > 2 == to big file")
    return all_dict


def sound_hash(h, from_M, to_N, source_file, target_dir):
    """
    create {hash : [w1, w2 .. ] }
    prevent to have only 1 word in list (word itself ... not similar sounding ... )
    """

    word_list = get_list_of_words(source_file=source_file)

    source_filename = ntpath.basename(source_file)
    target_filename = source_filename + "_" + h + "_" + str(from_M) + "_" + str(to_N) + ".pkl"

    assert h in ["SOUNDEX", "NYSIIS", "MRA", "METAPHONE"]

    assert from_M == 1, "do not use for M > 1: file will be to big ..."

    all_dict = create_hashed(word_list, from_M)
    cnt_elems = len(word_list)

    if to_N == 1:
        for w in word_list:
            hash = get_hash(w, h)
            if hash in all_dict:
                all_dict[hash].append(w.lower())

    elif to_N == 2:
        c = 0
        for w1 in word_list:
            for w2 in word_list:
                w = w1 + w2
                hash = get_hash(w, h)
                if hash in all_dict:
                    all_dict[hash].append(w1.lower() + " " + w2.lower())
            # Info ...
            c += 1
            if c>1 and c%100 == 0:
                print("{} / {}".format(c, cnt_elems))
    else:
        raise NotImplementedError("do not use for N > 2:  file will be to big ...")

    # !!! Remove single words !!!!
    for k, lst_similar in all_dict.items():
        if from_M == to_N:
            if len(lst_similar) <= 1:
                del all_dict[k]
        else:
            if len(lst_similar) == 0:
                del all_dict[k]

    # Write  it ...
    pickle_data(all_dict, target_dir, target_filename)

    # Debug statistic
    avg_similar = 0
    total = 0
    for key, value in all_dict.items():
        # print("w: {}   similar:{}   list: {}".format(key, len(value), value))
        avg_similar += len(value)
        total += 1

    print("average cnt similar {:.2f}".format(avg_similar /total))


if __name__ == "__main__":
    """
    Average hash matches for 1:1 words in words-400K (370+K most common English words)
    1:1       |  AVG_SIMILAR
    ------------------------
    SOUNDEX   | 62.13
    NYSIIS    |  1.70
    MRA       |  1.94  
    METAPHONE |  1.95
    
    
    Average hash matches for 1:2 words in words-30K (30K most common English words)
    1:2       | AVG_SIMILAR
    -----------------------
    SOUNDEX   |    inf
    NYSIIS    |  655.74
    MRA       |  1867.36 (quite a lot, right?)
    METAPHONE |  1568.13 (quite a lot, right?)
    """

    # Set what to do
    source_file = 'data/words-30K'
    hashes_to_use = ["METAPHONE", "MRA", "NYSIIS"]  # "SOUNDEX",
    mappings_to_use = [(1, 2)]  # (1, 1)

    for hash_to_use in hashes_to_use:
        for x in mappings_to_use:
            from_M, to_N = x
            sound_hash(source_file=source_file,
                       target_dir="hashtable",
                       h=hash_to_use,
                       from_M=from_M,
                       to_N=to_N)
