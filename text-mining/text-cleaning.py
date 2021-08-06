import string
import regex
import re
import nltk
from nltk.corpus import stopwords


nltk.download()
stop_en = stopwords.words("english")

# Example sentences
sent1 = "Sick times 🍫☕️💓#like #chocolate #coffee #iloveyou #coupleachievements"
sent2 = "Check out this https://google.com/foobar you will not believe it"
sent3 = ["i","have","a","cat","named","sir","purralot","but","he","does","not", "purr", "at", "all"]

# Removing punctuation
translator = str.maketrans('', '', string.punctuation)
sent_pun = sent1.translate(translator)
print(sent_pun)


# Removing emojis
emoji_pattern = regex.compile("""\p{So}\p{Sk}*""")
sent_pun_uni = emoji_pattern.sub(r' ', sent_pun)
print(sent_pun_uni)

# Removing urls
sent2 = re.sub(r"http\S+", "", sent2)
print(sent2)


# Removing stopwords
sent3 = [x for x in sent3 if not x in stop_en]
print(sent3)




