import re
import nltk

from foldoc_compsci_corpus import foldoc_dictionary_corpus

nltk.download('words')
from nltk.corpus import words
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import words

english_words = set(words.words())
stop_words = sw.words('english')


def get_compsi_words() -> list:
    compsci_words = []
    compsci_corpus = foldoc_dictionary_corpus()

    entry: str
    for entry in compsci_corpus:
        entry = entry.lower()

        if entry in compsci_words:
            continue

        compsci_words.append(entry)

    return compsci_words


compsci_words = set(get_compsi_words())
allowed_words = english_words.copy()
allowed_words.update(compsci_words)


def filter_allowed_words(text: str):
    text = re.sub(r'[^a-zA-Z0-9 -]', '', text)

    words = text.split()

    filtered_words = []

    for word in words:
        word = word.lower()

        if word in allowed_words and word not in stop_words:
            filtered_words.append(word)

    filtered_text = ' '.join(filtered_words)

    return filtered_text
