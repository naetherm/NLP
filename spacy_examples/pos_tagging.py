import nltk
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

def split_sentences_nltk(text):
    sentences = tokenizer.tokenize(text)
    return sentences

def split_sentences_spacy(text):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]

def pos_tag_spacy(text):
    doc = nlp(text)
    words = [token.text for token in doc]
    pos = [token.pos_ for token in doc]
    return list(zip(words, pos))

def pos_tag_nltk(text):
    words = tokenize_nltk(text)
    words_with_pos = nltk.pos_tag(words)
    return words_with_pos

def pos_tag(text):
    text = preprocess_text(text)
    words_with_pos = pos_tag_nltk(text)
    return words_with_pos

def main():
    random_text = read_file("./random_text.txt")
    random_text = preprocess_text(random_text)
    words_with_pos = pos_tag(random_text)
    print(words_with_pos)

if __name__ == '__main__':
    start = time.time()
    main()
