import time
import nltk
import spacy

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
nlp = spacy.load("en_core_web_sm")

def read_file(filename):
    file = open(filename, "r", encoding="utf-8") 
    return file.read()

def preprocess_text(text):
    text = text.replace("\n", " ")
    return text

def split_sentences_nltk(text):
    sentences = tokenizer.tokenize(text)
    return sentences

def split_sentences_spacy(text):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]

def split_sentences(text):
    return split_sentences_nltk(text)

def main():
    random_text = read_file("./random_text.txt")
    random_text = preprocess_text(random_text)
    sentences = split_sentences(random_text)
    print(sentences)

if __name__ == '__main__':
    start = time.time()
    main()
