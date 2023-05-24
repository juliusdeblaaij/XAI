import re
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def pre_process_text(text):
    text = re.sub('[^a-zA-Z0-9 -]', '', str(text))
    text = " ".join(str(text).split())
    text = text.lower()

    return text

def stem_text(text):
    words = text.split(' ')
    stemmed_text = ""

    for word in words:
        stemmed_text += " " + ps.stem(word)

    return stemmed_text.lstrip(' ')