'''
Based on "The Free On-line Dictionary of Computing, http://foldoc.org/, Editor Denis Howe"
'''

import re
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

foldoc_dictionary_path = 'foldoc-computing-dictionary.txt'
_foldoc_compsci_corpus = None


def foldoc_dictionary_corpus() -> list:
    global _foldoc_compsci_corpus, foldoc_dictionary_path, stop_words

    if _foldoc_compsci_corpus is not None:
        return _foldoc_compsci_corpus
    else:
        _foldoc_compsci_corpus = []

        with open(foldoc_dictionary_path) as f:
            lines = f.readlines()

            for line in lines:
                if check_of_foldoc_entry(line) == True:
                    clean_line = line.rstrip('\n')
                    clean_line = re.sub('[^a-zA-Z0-9 -\.]', '', str(clean_line))

                    if clean_line == "":
                        continue

                    if clean_line in stop_words:
                        continue

                    _foldoc_compsci_corpus.append(clean_line)

    return _foldoc_compsci_corpus


def check_of_foldoc_entry(string: str) -> bool:

    if string == "" or string.startswith(' ') or string.startswith('\n') or string.startswith('\r') or string.startswith('\t') or string[0].isdigit() :
        return False
    else:
        clean_string = string.rstrip('\n')
        clean_string = re.sub('[^a-zA-Z0-9 -]', '', str(clean_string))

        if clean_string == "" or clean_string.isnumeric():
            return False

        return True

if __name__ == "__main__":
    print(foldoc_dictionary_corpus())

