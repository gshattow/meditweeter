import re, string
from nltk.corpus import stopwords 

stop = set(stopwords.words('english'))

def simplify_string(s):
	s = re.sub(r'https?:\/\/.*[\r\n]*', ' ', s)
	s = "".join(i for i in s if ord(i)<128)
	s = re.sub('/', ' ', s)
	s = s.replace('\n', ' ')
	s = s.replace('\r', ' ')
	s = s.replace('[0-9]', '')
	table = string.maketrans("","")
	s = ''.join(s.encode('utf-8').translate(table, string.punctuation))
	return s

def simplify_string_db(s):
	s = "".join(i for i in s if ord(i)<128)
	s = s.replace('\n', ' ')
	s = s.replace('\r', ' ')
	s = s.replace("'", "''")
	s = s.replace('\\\\','\\')
	s = s.encode("ascii", "replace")
	return s


def make_ascii(s) :
	return "".join(i for i in s if ord(i)<128)
	
	
def go_words(s) :
	go_words = []
	for word in s.split() :
		if word not in stop : go_words.append(word)
	return go_words
	

def profile_to_words( raw_profile ) :
	simplify = simplify_string(raw_profile)
	simplify = simplify.lower()
	words = go_words(simplify)
	return ' '.join(words)

