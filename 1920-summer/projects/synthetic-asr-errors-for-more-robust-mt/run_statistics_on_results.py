import fileinput
import os
from jiwer import wer


def file_wer(file1, file2):
    with open(file1, 'r') as f:
        lines_1 = [line.rstrip() for line in f]
    with open(file2, 'r') as f:
        lines_2 = [line.rstrip() for line in f]

    assert len(lines_1) == len(lines_2), "Both files should have same number of lines ..."
    error_sum = 0
    for s1, s2 in zip(lines_1, lines_2):
        error_sum += wer(s1, s2)

    return error_sum / len(lines_1)


def wer_over_files(source_dir, target_dir, filename_list):

    with open(filename_list) as f:
        files_list = [line.rstrip() for line in f]

    list_of_source_filenames = [os.path.join(source_dir, f) for f in files_list]
    list_of_target_filenames = [os.path.join(target_dir, f) for f in files_list]

    assert len(list_of_source_filenames) == len(list_of_target_filenames), "We must have same number of lines for comparison"

    total_avg_wer = 0.0
    for file1, file2 in zip(list_of_source_filenames, list_of_target_filenames):
        w = file_wer(file1, file2)
        total_avg_wer += w
        print("WER:{:.2f}  FILE: {}".format(w, file2))

    print("total WER: {:.2f}".format(total_avg_wer / len(list_of_source_filenames)))


if __name__ == "__main__":
    source_dir = "example-input"
    target_dir = "example-output"
    filename_list = "input-files-list.txt"

    wer_over_files(source_dir, target_dir, filename_list)
