import fileinput
import logging
import os
from abc import abstractmethod

import logger


class SentenceGenerator:
    def __init__(self):
        self.logger = logging.getLogger(logger.BasicLogger.logger_name)

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass



# I do not have access to CzEng 2.0
# but I thing that code from list_files_generator can help...
class CzEngSentenceGenerator(SentenceGenerator):
    def __init__(self):
        SentenceGenerator.__init__(self)
        raise NotImplementedError

    def __next__(self):
        """
        HomoNoiserScript expects that generator returns tuple
        (document_name, sentence_id, line)
        document_name ... path to the current input document ("example-input/doc1.txt")
                      ... HomoNoiserScript uses the name to write the noisified sentence to the ("example-output/doc1.txt")
        sentence_id   ... currently unused
        line          ...  str == 1 line (== 1 sentence) from document_name
        """
        # todo - Implement this
        raise NotImplementedError

    def __iter__(self):
        return self


class SentencesFromListOfFiles(SentenceGenerator):
    def __init__(self, base_path, files_list):
        SentenceGenerator.__init__(self)
        self.doc_name = None
        self.sentence_id = None

        if not os.path.isdir(base_path):
            self.logger.error("Source directory not found: {}".format(base_path))
            raise FileNotFoundError

        if not os.path.isfile(files_list):
            self.logger.error("Source list not found: {}".format(files_list))
            raise FileNotFoundError

        with open(files_list) as f:
            self.files_list = [line.rstrip() for line in f]

        self.list_of_filenames = [os.path.join(base_path, f) for f in self.files_list]
        self.stream = fileinput.input(self.list_of_filenames)

    def __next__(self):
        line = next(self.stream)
        line = line.strip()
        curr_file = self.stream.filename()  # curret file name
        if self.doc_name is None or curr_file != self.doc_name:
            self.doc_name = curr_file
            self.logger.info("DOC_N:{} reading file & adding noise".format(self.doc_name))
        self.sentence_id = self.stream.filelineno()  # line in current file

        self.logger.debug("DOC_N:{} S_ID:{} S:'{}'".format(self.doc_name, self.sentence_id, line))
        return self.doc_name, self.sentence_id, line

    def __iter__(self):
        return self


class DebugGenerator(SentenceGenerator):
    def __init__(self):
        SentenceGenerator.__init__(self)
        self.doc_name = "-"
        self.sentence_id = -1
        self.sentences = ["Hello, my name is Peter.",
                          "George Floyd death: US cities order curfews amid widespread clashes!",
                          "Three other officers present at the time have also since been sacked",
                          "In New York, video showed a police car driving into a crowd of protesters.",
                          "San Francisco is the latest to impose a curfew, announced by Mayor London Breed for 20:00 "
                          "local time on Sunday, after looting and violence."]

    def __next__(self):
        if self.sentence_id + 1 < len(self.sentences):
            self.sentence_id += 1
            self.logger.debug(
                "DOC_N:{} S_ID:{} S:'{}'".format(self.doc_name, self.sentence_id, self.sentences[self.sentence_id]))
            return self.doc_name, self.sentence_id, self.sentences[self.sentence_id]
        else:
            raise StopIteration

    def __iter__(self):
        return self