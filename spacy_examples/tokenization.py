import nltk
from nltk.tokenize import TweetTokenizer
import time
import spacy

nlp = spacy.load("en_core_web_sm")

def read_file(filename):
    file = open(filename, "r", encoding="utf-8") 
    return file.read()

def preprocess_text(text):
    text = text.replace("\n", " ")
    return text

def tokenize_nltk(text):
    return nltk.tokenize.word_tokenize(text)

def tokenize_spacy(text):
    doc = nlp(text)
    return [token.text for token in doc]

def tokenize(text):
    return tokenize_spacy(text)

def main():
    random_text = read_file("./random_text.txt")
    random_text = preprocess_text(random_text)
    words = tokenize(random_text)
    print(words)

if __name__ == '__main__':
    start = time.time()
    main()
    print("%s s" % (time.time() - start))
