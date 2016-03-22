#!/usr/bin/python
import pandas as pd
import numpy as np
import re
import string
import functions as fn
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation

stop = set(stopwords.words('english'))

class Preprocess:

 
	def frequentist_score(self, profile) :
		# Replicate Epidemico's "first pass" by including 
		# profiles with dr/doctor/md
		words = profile.split()
		score = 0
		if len(words) > 0 :
			for word in words : 
				if word in ['dr', 'doctor', 'md'] : score += 1
		return score 

	def clean_profile(self, subset) :
		# Returns a list of profiles that have been "cleaned" of 
		# stop words, urls, punctuation, and capital letters
		clean_profiles = []
		for i in xrange( 0, len(subset) ):
			clean_profiles.append(str(fn.profile_to_words(subset[i])))
		return clean_profiles


	def vectorize_words(self, clean_profiles, max_features = 500) :
		# Vectorize the words in the cleaned profiles using 
		# term frequency/inverse document frequency (TF-IDF)
		print "Creating the bag of words...\n"

		# Initialize the "CountVectorizer" object, which is scikit-learn's
		# bag of words tool.  
		vectorizer = TfidfVectorizer(min_df=1, max_features = max_features) 
		vectorizer._validate_vocabulary()

		# fit_transform() does two functions: First, it fits the model
		# and learns the vocabulary; second, it transforms our training data
		# into feature vectors. The input to fit_transform should be a list of 
		# strings.
		data_features = vectorizer.fit_transform(clean_profiles)

		# Numpy arrays are easy to work with, so convert the result to an 
		# array
		data_features = data_features.toarray()
		print data_features.shape

		vocab = vectorizer.get_feature_names()

		# Sum up the counts of each vocabulary word
		dist = np.sum(data_features, axis=0)

	
		return vectorizer, data_features, vocab, dist

	def train_test(self, data, f_train = 0.75) :
		# Break list of tweeters into testing and training sets
		n_entries = data.shape[0]
		n_train = int(f_train*n_entries)
		train = data[:n_train]
		test = data[n_train:]

		print train.shape, 'profiles for training'
		print test.shape, 'profiles for testing'

		return train, test

class Modelling :
	def train_random_forest(self, vectorizer, train_data_features, train_class) :
		# Train the model using a random forest
		print "Training the random forest..."

		# Initialize a Random Forest classifier with 100 trees
		forest = RandomForestClassifier(n_estimators = 100) 

		# Fit the forest to the training set, using the bag of words as 
		# features and the sentiment labels as the response variable
		#
		# This may take a few minutes to run
		vec_forest = Pipeline([('vec', vectorizer), ('rf', forest)])
		# save forest and vectorizer for later
		joblib.dump(vec_forest, 'forest/forest.pkl') 

		forest = forest.fit( train_data_features, train_class )

		importances = forest.feature_importances_
		scores = cross_validation.cross_val_score(forest,
			train_data_features, train_class, cv=5)
		print 'Cross validation scores:', scores.mean(), '+/-', scores.std()

		return forest, importances



	def test_set(self, vectorizer, forest, clean_test_profiles) :
		print 'Running model on test set...'
		# run forest model on the test set to test model
		# and find first pass scores
	
	
		# Get a bag of words for the test set, and convert to a numpy array
		test_data_features = vectorizer.transform(clean_test_profiles)
		test_data_features = test_data_features.toarray()


		# Use the random forest to make sentiment label predictions
		model_class = forest.predict(test_data_features)

		probs = forest.predict_proba(test_data_features)[:,1]
	
		return model_class, probs

class ReadWrite :
	def read_tweeters(self, file) :
		# Read in the list of profiles, which has been randomly shuffled
		# and marked if a medical professional or not (1 or 0, respectively)
		data = pd.read_csv(file, header=0, \
			delimiter="\t")
					
		print data.shape
		print data.columns.values
		return data


	def write_words(self, file, vocab, amount) :
		# For each, print the vocabulary word and the number of times it 
		# appears in the training set
		f = open(file, 'w')
		for tag, count in zip(vocab, amount):
			f.write('{0:f}'.format(count) + ',' + tag + '\n')
		f.close()
		return

	def write_to_file(self, test_class, model_class, freq_scores, probs, profiles) :
		# Copy the test set to a pandas dataframe including actual class,
		# the "first pass" model, the probability (used to plot the ROC curve),
		# and the original profile 
		output = pd.DataFrame( data={"actual_sentiment":test_class,
			"model_sentiment":model_class, "freq_score" : freq_scores, 
			"probability" : probs, "profile":profiles} )

		# Use pandas to write the comma-separated output file
		output.to_csv( "Bag_of_Words_model.csv", index=False)#, quoting=3 )

		return

if __name__ == '__main__':

	rw = ReadWrite()
	mod = Modelling()
	pre = Preprocess()

	#Load the tweets
	file = 'shuffled_tweeters.tsv'
	data = rw.read_tweeters(file) 
	
	# Break tweets into train/test sets of f_train, 1-f_train sizes
	train, test = pre.train_test(data, f_train = 0.75)
	
	# Save df as array (easier to deal with later)
	train_class = train["sentiment"].values
	train_profiles = train["profile"].values
	test_class = test["sentiment"].values
	test_profiles = test["profile"].values
	
	# Clean training profiles - take out stop words, punctuation, etc.
	clean_train_profiles = pre.clean_profile(train_profiles)
	
	# Vectorize training profiles using TF-IDF
	vectorizer, train_data_features, vocab, dist =\
		pre.vectorize_words(clean_train_profiles, max_features = 500)
	
	# Save word counts to file (not totally necessary, but interesting)
	rw.write_words('word_counts.txt', vocab, dist)
	
	# Train the model! 
	forest, importances = mod.train_random_forest(vectorizer,\
		train_data_features, train_class)
		
	# Write word importances to file
	rw.write_words('word_importance.txt', vocab, importances)
	
	# Preprocess test set of profiles
	clean_test_profiles = pre.clean_profile(test_profiles)
	
	# Run test set through model
	model_class, probs = mod.test_set(vectorizer, forest, clean_test_profiles)
	
	# Find frequentist score for test profiles (Epidemico first pass)
	freq_scores = [pre.frequentist_score(profile) for profile in clean_test_profiles]
	
	# Write test set results to file
	rw.write_to_file(test_class, model_class, freq_scores, probs, test_profiles)