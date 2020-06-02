import jellyfish
import ntpath
import time


from train_dictionary.train_util import get_list_of_words, pickle_data


def mra_1_to_all(word, all_words, threshold):
    similar_list = []
    for j, w2 in enumerate(all_words):
        if word == w2:  # skip -- same word
            continue

        # Must similar according to Match Rating Comparison (similarity on MRA hashes)
        if jellyfish.match_rating_comparison(word, w2):
            # And also score must be higher than threshold
            if jellyfish.jaro_winkler_similarity(word, w2) >= threshold:
                similar_list.append(w2)

    return similar_list


def mra_dictionary(source_file, dir_name, threshold, save_every=10000):

    # 1. Load all words
    word_list = get_list_of_words(source_file=source_file)
    total_len = len(word_list)

    start = time.time()
    all_dict = {}

    source_filename = ntpath.basename(source_file)
    target_filename = source_filename + "_mra_" + str(threshold) + ".pkl"

    # 2. Align them ...
    for src_id, w1 in enumerate(word_list):
        # Info print ...
        if src_id > 0 and (src_id + 1) % 100 == 0:
            t = time.time() - start
            start = time.time()
            tt = (total_len - src_id + 1) / 100
            tt = t * tt
            hod = tt / 60 / 60
            print("{}/{} .. {:.1f} .. remaining {:.1f} h".format(src_id + 1, total_len, t, hod))

        # Compare current word to all others, if some are similar, append the dictionary
        similar_list = mra_1_to_all(w1, word_list, threshold)
        if len(similar_list) > 0:
            all_dict[w1] = similar_list

        # save data to prevent loss...
        if src_id > 0 and src_id % save_every == 0:
            pickle_data(all_dict, dir_name, target_filename)

    # save at the end ...
    pickle_data(all_dict, dir_name, target_filename)


if __name__ == "__main__":
    """
    Create dictionary in the specified format
    We use the  Match Rating Approach  (similarity on MRA hash) + Jaro-Winkler similarity 
    """

    # Parameters
    source_file = 'data/words-400K'
    target_directory = "dictionary"
    threshold = 0.8  # Words w1 w2 considered similar if Jaro-Winkler( w1, w2 ) >=  threshold
    save_every = 10000  # to prevent loosing the data ...

    mra_dictionary(source_file=source_file,
                   dir_name=target_directory,
                   threshold=threshold,
                   save_every=save_every)
