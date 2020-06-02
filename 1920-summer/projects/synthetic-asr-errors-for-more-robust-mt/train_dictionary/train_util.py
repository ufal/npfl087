import os
import pickle


def pickle_data(obj, target_dir, filename):
    if not os.path.exists(target_dir):
        print("Target dir: '{}' not found, creating it...".format(target_dir))
        os.makedirs(target_dir)

    path = os.path.join(target_dir, filename)

    print("Writing pickle: {}".format(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_list_of_words(source_file):
    words = []
    with open(source_file, 'r') as f:
        for line in f:
            word = line.strip().lower()
            words.append(word)
    print("All words loaded from {}".format(source_file))
    return words
