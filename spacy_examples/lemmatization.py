import nltk
from nltk.stem import WordNetLemmatizer

def read_file(filename):
    file = open(filename, "r", encoding="utf-8") 
    return file.read()

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

lemmatizer = WordNetLemmatizer()
pos_mapping = {'JJ':'a', 'JJR':'a', 'JJS':'a', 'NN':'n', 'NNS':'n', 'VBD':'v', 'VBG':'v', 'VBN':'v', 'VBP':'v', 'VBZ':'v'}
accepted_pos = {'a', 'v', 'n'}

def lemmatize_text(text):
    words = pos_tag_nltk(text)
    words = [(word_tuple[0], pos_mapping[word_tuple[1]] if word_tuple[1] in pos_mapping.keys() else word_tuple[1]) for word_tuple in words]
    words = [(lemmatizer.lemmatize(word_tuple[0]) if word_tuple[1] in accepted_pos else word_tuple[0], word_tuple[1]) for word_tuple in words]
    return words


def main():
    sentences = [
        'Where did he learn to dance like that?',
        'His eyes were dancing with humor.',
        'She shook her head and danced away',
        'Alex was an excellent dancer.']
    output = []
    for sentence in sentences:
        output.append(" ".join([lemmatizer.lemmatize(i) for i in sentence.split()]))
    print(output)
    
    random_text = read_file("./random_text.txt")
    lem_text = lemmatize_text(random_text)
    print(lem_text)

if (__name__ == '__main__'):
    main()

