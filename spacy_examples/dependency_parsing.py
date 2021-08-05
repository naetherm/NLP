import spacy

sentences = ['Started his hearted any civilly.',
    'So me by marianne admitted speaking.',
    'Men bred fine call ask.',
    'Cease one miles truth day above seven.',
    'Suspicion sportsmen provision suffering mrs saw engrossed something.'
    'Snug soon he on plan in be dine some. ']
nlp = spacy.load('en_core_web_sm')

def get_dependency_parse(sentence):
    doc = nlp(sentence)
    for token in doc:
        print(token.text, "\t", token.dep_, "\t", spacy.explain(token.dep_))

def main():
    for i in range(len(sentences)):
        print("---")
        get_dependency_parse(sentences[i])

if (__name__ == "__main__"):
    main()
