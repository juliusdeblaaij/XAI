from nltk import PorterStemmer, sent_tokenize

from foldoc_compsci_corpus import foldoc_dictionary_corpus

from nltk.corpus import stopwords as sw
from inflector.languages.english import *
english_inflector = English()

from nltk.corpus import wordnet

wordnet_full = wordnet.all_lemma_names()
english_words = []
for lemma in wordnet_full:
    if '_' in lemma:
        continue

    english_words.append(lemma)

english_words = set(english_words)

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
custom_words = {"xusp", "xdnn", "this", "is", "75%", "90%"}
allowed_words = english_words.copy()
allowed_words.update(compsci_words)
allowed_words.update(custom_words)

porter_stemmer = PorterStemmer()

def filter_allowed_words(old_text: str, keep_stop_words=False):
    text = old_text.lower()
    text = text.replace(r'\n', ' ')
    text = text.replace(r'. ', ' ')
    text = text.replace(r'; ', ' ')
    text = text.replace(r'! ', ' ')
    text = text.replace(r'\t', ' ')
    text = text.replace(r'\r', ' ')
    text = text.replace(r',', ' ')

    words = []
    for word in text.split():
        if word not in allowed_words:
            word = re.sub('[^a-zA-Z0-9]', '', str(word))
        words.append(word)

    filtered_words = []

    for word in words:
        if word in ["this", "is"]:
            singularized_word = word
            stemmed_word = word
        else:
            singularized_word = english_inflector.singularize(word=word)
            stemmed_word = porter_stemmer.stem(word)

        if not keep_stop_words:
            if singularized_word in allowed_words and singularized_word not in stop_words:
                filtered_words.append(word)
            elif stemmed_word in allowed_words and stemmed_word not in stop_words:
                filtered_words.append(word)
        else:
            if singularized_word in allowed_words or singularized_word in stop_words:
                filtered_words.append(word)
            elif stemmed_word in allowed_words or stemmed_word in stop_words:
                filtered_words.append(word)

    filtered_text = ' '.join(filtered_words)

    return filtered_text

def filter_allowed_words_in_sentences(old_sentences: list, keep_stop_words=False):
    new_sentences = []

    for sentence in old_sentences:
        sentence = sentence.lower()
        filtered_sentence = filter_allowed_words(sentence, keep_stop_words)
        filtered_sentence = filtered_sentence.capitalize()
        filtered_sentence += "."
        new_sentences.append(filtered_sentence)

    return new_sentences


if __name__ == "__main__":
    case = """XUSP is an algorithm that is designed to automatically determine the amount of story points for a given user story.
The prediction of the story point value is made using a specific process called xDNN.
xDNN classifies items based on the similarity between it and already learned items.
Prediction:
"As a follow up to xd-2877 experiment with the removal of the list of modules from basedefinition and reparse as needed. branch is here: https://github.com/pperalta/spring-xd/tree/deploy-refactor-2" is a user story, and is worth 8 story points. This was predicted because it is most similar to: 
 "Scd developer id like add support different binder types modules channels plug rabbit kafka source sink read write respectively" and similar to "Field engineer id like reference architectures built spring use reference building pocs scope get raw domain specific ideas first step" and similar to "Developer id like job module bootstrapped job launched shut once complete instead current behavior bootstrapping context module deployed regardless used achieve better resource utilization" (which are all user stories with a worth of 8 story points)."""
    sentences_in_case = sent_tokenize(case)
    for sentence in filter_allowed_words_in_sentences(sentences_in_case, keep_stop_words=True):
        print(sentence)