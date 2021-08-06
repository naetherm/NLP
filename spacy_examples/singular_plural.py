import nltk
from nltk.stem import WordNetLemmatizer
import inflect
import sys
import os


def tokenize_nltk(text):
    return nltk.tokenize.word_tokenize(text)

def tokenize_spacy(text):
    doc = nlp(text)
    return [token.text for token in doc]

def pos_tag_spacy(text):
    doc = nlp(text)
    words = [token.text for token in doc]
    pos = [token.pos_ for token in doc]
    return list(zip(words, pos))

def pos_tag_nltk(text):
    words = tokenize_nltk(text)
    words_with_pos = nltk.pos_tag(words)
    return words_with_pos
    
def read_file(filename):
    file = open(filename, "r", encoding="utf-8") 
    return file.read()

def preprocess_text(text):
    text = text.replace("\n", " ")
    return text

def is_plural(noun):
    wnl = WordNetLemmatizer()
    lemma = wnl.lemmatize(noun, 'n')
    print(lemma)
    plural = True if noun is not lemma else False
    return plural

def is_singular(noun):
    return not is_plural(noun)

def get_plural_form(singular_noun):
    p = inflect.engine()
    return p.plural(singular_noun)
    
def get_singular_form(plural_noun):
    p = inflect.engine()
    plural = p.singular_noun(plural_noun)
    if (plural):
        return plural
    else:
        return plural_noun

def filter_nouns(words_with_pos):
    noun_set = ["NN", "NNS"]
    nouns = [word for word in words_with_pos if word[1] in noun_set]
    return nouns

def plurals_wn(words_with_pos):
    other_nouns = []
    for noun_info in words_with_pos:
        word = noun_info[0]
        plural = is_plural(word)
        if (plural):
            singular = get_singular_form(word)
            other_nouns.append(singular)
        else:
            plural = get_plural_form(word)
            other_nouns.append(plural)
    return other_nouns

def plurals_nltk(nouns):
    other_nouns = []
    for noun_info in nouns:
        word = noun_info[0]
        pos = noun_info[1]
        if (pos == "NNS"):
            singular = get_singular_form(noun_info[0])
            other_nouns.append(singular)
        else:
            plural = get_plural_form(noun_info[0])
            other_nouns.append(plural)
    return other_nouns

def main():
    random_text = read_file("./random_text.txt")
    random_text = preprocess_text(random_text)
    words_with_pos = pos_tag_nltk(random_text)
    nouns = filter_nouns(words_with_pos)
    print(nouns)
    other_nouns_wn = plurals_wn(nouns)
    print(other_nouns_wn)
    other_nouns_nltk = plurals_nltk(nouns)
    print(other_nouns_nltk)
    
    # In here we find the problem that the first method will return false
    # because the lemmatizer does not correctly inflect the word "men"
    print(is_plural("men"))
    print(is_singular("men"))



if (__name__ == "__main__"):
    main()
