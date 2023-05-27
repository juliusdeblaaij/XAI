import re

def pre_process_text(text):
    text = re.sub('[^a-zA-Z0-9 -]', '', str(text))
    text = " ".join(str(text).split())
    text = text.lower()

    return text

