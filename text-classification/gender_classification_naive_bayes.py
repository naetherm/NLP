import nltk
from nltk.corpus import names
import random

SET_SPLIT = 300

names = (
    [(name, 'male') for name in names.words('male.txt')] +
    [(name, 'female') for name in names.words('female.txt')])
    
random.shuffle(names)

def single_features(word):
    return {
        'last_letter': word[-1]
    }
    
feature_sets = [(single_features(n), gender) for (n, gender) in names]
print(len(feature_sets))


classifier = nltk.NaiveBayesClassifier.train(feature_sets[SET_SPLIT:])

print(classifier.classify(single_features('Markus')))
print(classifier.classify(single_features('Laura')))

print(nltk.classify.accuracy(classifier, feature_sets[:SET_SPLIT]))


def more_features(word):
    return {
        'last_letter': word[-1],
        'second_last_letter': word[-2],
        'rest': word[:-2]
    }
    
feature_sets = [(more_features(n), gender) for (n, gender) in names]
print(len(feature_sets))


classifier = nltk.NaiveBayesClassifier.train(feature_sets[SET_SPLIT:])

print(classifier.classify(more_features('Markus')))
print(classifier.classify(more_features('Laura')))

print(nltk.classify.accuracy(classifier, feature_sets[:SET_SPLIT]))
