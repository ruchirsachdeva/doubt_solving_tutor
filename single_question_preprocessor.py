import json
import re

import numpy as np
from nltk.stem.porter import PorterStemmer

maxSeqLength = 30
ps = PorterStemmer()


def get_zipped_questions(question):
    print('CLEANING, TOKENIZING AND STEMMING THE INPUT QUESTION')
    # file_path = sys.argv[1]
    question1 = []
    question1.append(get_words(question))
    # zipped_object = list(zip(question1))
    # with open('stemmed_split_sentences','w') as myfile:
    # 	print(json.dump(zipped_object, myfile))
    return zip(question1)


def get_words(q1):
    question1_cleaned = clean_text(q1.lower())
    question1_words = question1_cleaned.split()
    question1_words = [ps.stem(word) for word in question1_words]
    return question1_words


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# The function below converts text into vectors of words
def vectorize(question):
    print('VECTORIZING THE INPUT OF THE TWO QUESTIONS')
    wordlist = []
    known = 0
    unkown = 0
    with open('wordlist', 'r') as myfile:
        wordlist = myfile.readlines()
        wordlist = [word.lower().strip() for word in wordlist]
    zipped_data = get_zipped_questions(question)
    number_of_examples = len(zipped_data)
    question_one_ids = np.zeros((number_of_examples, maxSeqLength), dtype='int32')
    example_counter = 0

    for question1_words in zipped_data:
        wordcounter = 0
        '''
        print question1_words
        print example_counter
        '''
        for word in question1_words:
            try:
                question_one_ids[example_counter][wordcounter] = wordlist.index(word)
                known += 1
                wordcounter += 1
            except ValueError:
                question_one_ids[example_counter][wordcounter] = 3999999
                wordcounter += 1
                unkown += 1
        example_counter += 1

    np.save('input_ids_matrix', question_one_ids)
    print(known)
    print(unkown)
