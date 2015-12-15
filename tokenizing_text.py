from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict

sentence = "Peter piper picked a peck of pickled peppers. \
        A peck of pickled peppers, Peter Piper picked !!! \
        If Peter Piper picked a peck of pickled peppers, \
        Whereas the peck of pickled peppers Peter Piper \
        picked ?"

sent_list = sent_tokenize(sentence)

print "# Sentences = %d"%(len(sent_list))
print "Sentences"
for sent in sent_list: print sent

word_dict = defaultdict(list)
for i, sent in enumerate(sent_list):
    word_dict[i].extend(word_tokenize(sent))

print word_dict

