import pandas as pd
from nltk.stem.porter import PorterStemmer
import re
import numpy as np
import json
import sys
import collections
maxSeqLength = 30
ps = PorterStemmer()
def load_zipped_questions():
	with open('stemmed_split_sentences','r') as myfile:
		data = json.load(myfile)
	return data

def save_zipped_questions():
	file_path = 'C:\\dev\\projects\lnu\\thesis\\DOUBT_SOLVING_TUTOR\\data\\quora_duplicate_questions.tsv'
	print('CLEANING, TOKENIZING AND STEMMING THE TRAINING DATASET')
	#file_path = sys.argv[1]
	csv_dataframe  = pd.read_csv(file_path, sep='\t', encoding='utf-8')
	csv_dataframe = csv_dataframe[['question1','question2','is_duplicate']]
	question1 = []
	question2 = []
	is_duplicate = []
	for index, row in csv_dataframe.iterrows():
		question1_words = get_words(str(row['question1']))
		question2_words = get_words(str(row['question2']))
		# Skip questions with words more than 30.
		if len(question1_words)>30 or len(question2_words)>30:
			pass
		else:
			question1.append(question1_words)
			question2.append(question2_words)
			is_duplicate.append(row['is_duplicate'])
	zipped_object = list(zip(question1,question2,is_duplicate))
	with open('stemmed_split_sentences','w') as myfile:
		print(json.dump(zipped_object, myfile))
	
	return zip(question1,question2,is_duplicate)


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

#The function below converts text into vectors of words
def vectorize():
	print('VECTORIZING QUESTIONS')
	wordlist = []
	known = 0
	unkown = 0
	with open('wordlist','r') as wordlistfile:
		wordlist = wordlistfile.readlines()
		wordlist = [word.lower().strip() for word in wordlist]
	zipped_data = load_zipped_questions()
	number_of_examples = len(zipped_data)	
	question_one_ids = np.zeros((number_of_examples, maxSeqLength), dtype='int32')
	question_two_ids = np.zeros((number_of_examples, maxSeqLength), dtype='int32')
	is_duplicate = np.zeros((number_of_examples,1),dtype = 'int32')
	example_counter = 0
	
	for question1_words,question2_words,is_duplicate in zipped_data:
		wordcounter = 0
		for word in question1_words:
			try:
				question_one_ids[example_counter][wordcounter] = wordlist.index(word)
				known+=1
				wordcounter+=1
			except ValueError:								     
				question_one_ids[example_counter][wordcounter] = 3999999 
				wordcounter+=1
				unkown+=1
		wordcounter = 0
		for word in question2_words:
			try:
				question_two_ids[example_counter][wordcounter] = wordlist.index(word)
				known+=1
				wordcounter+=1
			except ValueError:								     
				question_two_ids[example_counter][wordcounter] = 3999999 
				wordcounter+=1
				unkown+=1
		example_counter+=1
		if example_counter % 100 == 0:
			print(' ')
			print('NUMBER OF SENTENCE PAIRS DONE === ' + str(example_counter))
			print('TOTAL NUMBER OF SENTENCES LEFT   === ' + str(number_of_examples - example_counter))
			print(' ')
		if example_counter % 2000 == 0 :
			print(' SAVING THE COMPUTED VECTORS AT STEP == ' + str(example_counter))
			np.save('q1_ids_matrix',question_one_ids)
			np.save('q2_ids_matrix',question_two_ids)

		wordcounter = 0
	np.save('q1_ids_matrix',question_one_ids)
	np.save('q2_ids_matrix',question_two_ids)
	print(known)
	print(unkown)


def generate_target_values_array():
	zipped_object = load_zipped_questions()
	number_of_examples = len(zipped_object)
	is_same_matrix = np.zeros((number_of_examples,1), dtype='int32')
	example_counter = 0
	for _,_,is_duplicate in zipped_object:
		is_same_matrix[example_counter] = int(is_duplicate)
		example_counter += 1 
	np.save('is_same_matrix',is_same_matrix)

save_zipped_questions()
vectorize()
generate_target_values_array()
