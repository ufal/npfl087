#
# from pathlib import Path
#
# def get_project_root() -> Path:
#     """Returns project root folder."""
#     return Path(__file__).parent.parent
from train_dictionary.dictionary_from_scrape import parse_akward_file

if __name__ == "__main__":
    """
    Create dictionary of similar words in specified format.
    Source is list files
    Line in file == similar words with '/' delimiter
    """
    parse_akward_file("dictionary/mra_0.8_140K.txt", "/")
    exit(33)
