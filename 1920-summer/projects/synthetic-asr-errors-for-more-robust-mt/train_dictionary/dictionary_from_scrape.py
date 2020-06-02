from train_dictionary.train_util import pickle_data


def write_dict_to_file(filename, dict_):
    """
    This function is only for 'visual' debug. It creates text file representing the dictionary.
    """
    print("writing dict to file")
    with open(filename, "w") as file:
        for key, val in dict_.items():  # for each set of homonyms
            to_write = key + ": "
            # check that there is at least 2 homonyms...
            if len(val) < 1:
                print("warning: only 1 word: {}".format(val))
                continue

            # create one line
            for i, word in enumerate(val):
                if len(word) == 0:
                    print("warning: empty word")
                    continue

                to_write += word
                if i + 1 < len(val):
                    to_write += "/"

            # save line to file
            if len(to_write) > 0:
                to_write += '\n'
                file.write(to_write)
            else:
                print("warning: empty line")


def parse_single_file(file, delimiter):
    print("Parsing: {}".format(file))

    list_of_lines = []
    with open(file, "r") as f:  # for each line
        for line in f:
            line = line.strip()
            line = line.replace("!", "").replace("’", "'").lower().strip()  # extra normalisation...
            line = line.split(delimiter)
            for num in ["1", "2", "3", "4"]:
                if num in line:
                    line.remove(num)

            for t, wrd in enumerate(line):
                line[t] = wrd
                if not wrd.isalpha() and not wrd.replace("'", "").replace("-", "").replace(" ", "").replace(".",
                                                                                                            "").isalpha():
                    if len(wrd) == 0:
                        print("000: {}".format(line))
                    else:
                        print("'{}'".format(wrd))

            if len(line) > 1:
                list_of_lines.append(line)
    return list_of_lines


def merge_all_files(filenames, delimiter):
    lines_as_sets = []

    # read all parsed files
    for file in filenames:
        list_of_lines = parse_single_file(file, delimiter)
        for line in list_of_lines:
            new_set = set(line)  # read homonym group
            if len(new_set) == 1:  # skip when only 1 word == no homonyms
                continue

            added = False
            for i, set_line in enumerate(lines_as_sets):
                if set_line.intersection(new_set):
                    added = True
                    set_line = set_line.union(new_set)
                    lines_as_sets[i] = set_line
            if not added:
                lines_as_sets.append(new_set)

    total_unique_before = set({})
    for x in lines_as_sets:
        for w in x:
            total_unique_before.add(w)
    print("# unique words from all files: {}".format(len(total_unique_before)))

    # Do closure over all files
    lines_as_sets = do_transitive_closure(lines_as_sets)

    # back to list
    for i, set_line in enumerate(lines_as_sets):
        lines_as_sets[i] = sorted(list(set_line))

    return lines_as_sets


def do_transitive_closure(list_sets):
    # Transitive closure over sets
    print("Creating closure")
    while True:
        change = False
        for i, set_1 in enumerate(list_sets):
            for j, set_2 in enumerate(list_sets):
                if set_1.intersection(set_2) and set_1 != set_2:
                    change = True
                    union = set_1.union(set_2)
                    list_sets.remove(set_1)
                    list_sets.remove(set_2)
                    list_sets.append(union)
                    break
            if change:
                break

        if not change:
            break
    return list_sets


if __name__ == "__main__":
    """
    Create dictionary of similar words in specified format.
    Source is list files
    Line in file == similar words with '/' delimiter
    """

    import resource

    rsrc = resource.RLIMIT_DATA
    print(rsrc)
    soft, hard = resource.getrlimit(rsrc)
    print('Soft limit starts as  : {}'.format(hard) )
    exit(11)
    #
    # resource.setrlimit(rsrc, (1024, hard))  # limit to one kilobyte
    #
    # soft, hard = resource.getrlimit(rsrc)
    # print
    # 'Soft limit changed to :', soft

    delim = '/'
    fnames = ["data/homonyms-scraped" + "/" + str(i + 1) + ".txt" for i in range(8)]
    target_fname = "scraped-dict.pkl"
    target_dir = "dictionary"

    # Merge files
    list_list_similar = merge_all_files(fnames, delim)

    # Create dictionary in specified format
    fina_dict = {}
    # we want key in the dict for each of the group of similar words ...
    # eg.  list_similar = [dog, doggo, doggie]
    #                   --> { dog : [dogo, doggie]  , doggo: [dog, doggie] , doggie : [dog, doggo] }
    for list_similar in list_list_similar:
        for i in range(len(list_similar)):
            n = [x for x in list_similar if x != list_similar[i]]
            fina_dict[list_similar[i]] = n

    # write_dict_to_file("scraped-merged.txt", fina_dict)
    pickle_data(fina_dict, target_dir, target_fname)
