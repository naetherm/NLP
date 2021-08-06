from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

def stem_word(word):
    stem = stemmer.stem(word)
    return stem

def main():
    sentences = [
        'Where did he learn to dance like that?',
        'His eyes were dancing with humor.',
        'She shook her head and danced away',
        'Alex was an excellent dancer.']
    output = []
    for sentence in sentences:
        output.append(" ".join([stem_word(i) for i in sentence.split()]))
    print(output)

if (__name__ == '__main__'):
    main()
