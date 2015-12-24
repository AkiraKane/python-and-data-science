from nltk.corpus import movie_reviews
from sklearn.cross_validation import StratifiedShuffleSplit
import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

def get_data():
    """
    Get movie review data
    """
    dataset = []
    y_labels = []

    #extract cateogires
    for cat in movie_reviews.categories():
        #for files in each category
        for fileid in movie_reviews.fileids(cat):
            #get the words in that category
            words = list(movie_reviews.words(fileid))
            dataset.append((words, cat))
            y_labels.append(cat)
    return dataset,y_labels

def get_train_test(input_dataset, ylabels):
    """
    Prepare a stratified train and test split
    """
    train_size = 0.7
    test_size = 1 - train_size
    stratified_split = StratifiedShuffleSplit(ylabels, test_size=test_size,
            n_iter=1, random_state=77)

    for train_indx, test_indx in stratified_split:
        train = [input_dataset[i] for i in train_indx]
        train_y = [ylabels[i] for i in train_indx]

        test = [input_dataset[i] for i in test_indx]
        test_y = [ylabels[i] for i in test_indx]

    return train,test,train_y,test_y

def build_word_features(instance):
    """
    Build feature dictionary
    Features are binary, name of the feature is word itself
    and value is 1. Features are stored in a dictionary
    called feature_set"""
    # Dictionary tos tore the featueres
    feature_set = {}
    # The first item in the instance tuple the word list
    words = instance[0]
    #populate feature dictionary
    for word in words:
        feature_set[word] = 1
    return (feature_set, instance[1])

def build_negate_features(instance):
    """
    If a word is preceeded by either 'not' or 'no'
    this functions adds a prefix 'Not_' to that word.
    It will also not inser tthe previous negation word
    'not' or 'no' in the feature dictionary
    """
    # Retrieveew words, first item in instance tuple
    words = instance[0]
    final_words = []
    # A boolean variable to track if the previous word is negation word
    negate = False
    # list of negation words
    negate_words = ['no', 'not']
    # On looping throughout the words, on encourtering 
    #a negation word, variable negate is set to True
    # negation wor dis not added to feature dicitonary
    # if negate variable is set to true
    # 'Not_' prefix is added to the word
    for word in words:
        if negate:
            word = 'Not_' + word
            negate = False
        if word not in negate_words:
            final_words.append(word)
        else:
            negate = True
    #Feature dictionary
    feature_set = {}
    for word in final_words:
        feature_set[word] = 1
    return (feature_set, instance[1])

def remove_stop_words(in_data):
    """
    Utility funciton to remove stop words
    from the given lsit of words
    """
    stopword_list = stopwords.words('english')
    negate_words = ['no', 'not']
    # we dont want to remove the negate words
    # hence we create a new stop word lsit exlcuding 
    # the negate words
    new_stopwords = [word for word in stopword_list if word not in negate_words]
    label = in_data[1]
    #remove stop words
    words = [word for word in in_data[0] if word not in new_stopwords]
    return (words, label)

def build_keyphrase_features(instance):
    """
    A function to extract key phrases for the given text.
    Key phrases are words of importance according to a measure
    In this key our phrase is of our length 2, i.e. two words in bigrams
    """
    feature_set = {}
    instance = remove_stop_words(instance)
    words = instance[0]

    bigram_finder = BigramCollocationFinder.from_words(words)
    # we use the raw frequency count of bigrams, ie.e bigrams
    # are ordered by their frequency of occurence in the descending order
    # and top 400 bigrams are selected
    bigrams = bigram_finder.nbest(BigramAssocMeasures.raw_freq, 400)
    for bigram in bigrams:
        feature_set[bigram] = 1
    return (feature_set, instance[1])

def build_model(features):
    """
    Build a naive bayes model
    with the given feature set.
    """
    model = nltk.NaiveBayesClassifier.train(features)
    return model

def probe_model(model, features, dataset_type='Train'):
    """
    A utility function to check the goodness of our model
    """
    accuracy = nltk.classify.accuracy(model, features)
    print "\n" + dataset_type + " Accuracy = %0.2f"%(accuracy*100) + "%"

def show_features(model, no_features=5):
    """
    A utility funciton to see how important a various features are for our model
    """
    print "\nFeature Importance"
    print "===================\n"
    print model.show_most_informative_features(no_features)

def build_model_cycle_1(train_data, dev_data):
    """
    First pass at trying out our model
    """
    train_features = map(build_word_features, train_data)
    dev_features = map(build_word_features, dev_data)
    model = build_model(train_features)
    probe_model(model, train_features)
    probe_model(model, dev_features, 'Dev')

    return model

def build_model_cycle_2(train_data, dev_data):
    """
    Second pass at trying out oru model
    """
    train_features = map(build_negate_features, train_data)
    dev_features = map(build_negate_features, dev_data)
    model = build_model(train_features)

    probe_model(model, train_features)
    probe_model(model, dev_features, 'Dev')

    return model

def build_model_cycle_3(train_data, dev_data):
    """
    Third pass attrying out our mdoel
    """

    train_features = map(build_keyphrase_features, train_data)
    dev_features = map(build_keyphrase_features, dev_data)
    model  = build_model(train_features)

    probe_model(model, train_features)
    probe_model(model, dev_features, 'Dev')

    test_features = map(build_keyphrase_features, test_data)
    probe_model(model, test_features, 'Test')

    return model

if __name__ == "__main__":
    #load data
    input_dataset, y_labels = get_data()

    # Train Data
    train_data, all_test_data, train_y, all_test_y = get_train_test(input_dataset, y_labels)

    # dev data
    dev_data, test_data, dev_y, test_y = get_train_test(all_test_data,
            all_test_y)

    # Look at the data size in our different datasets
    print "\nOriginal Data Size = ", len(input_dataset)
    print "\nTraning Data Size = ", len(train_data)
    print "\nDev Data Size = ", len(dev_data)
    print "\nTesting Data Size = ", len(test_data)


    # different passes on our model building exercise
    model_cycle_1 = build_model_cycle_1(train_data, dev_data)
    show_features(model_cycle_1)

    model_cycle_2 = build_model_cycle_2(train_data, dev_data)
    show_features(model_cycle_2)

    model_cycle_3 = build_model_cycle_3(train_data, dev_data)
    show_features(model_cycle_3)

