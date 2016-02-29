# meditweeter
The code and data behind meditweeter.com

Notes:

config.py needs the keys - I haven't put mine in for obvious reasons, but you can get your own from Twitter.

shuffled_tweeters.tsv is a list of twitter profiles accumulated through members.py and then shuffled. Feel free to use this list rather than fetch your own if you do not feel like getting keys from Twitter.

bag_of_words.py is the script that creates the actual model. It requires shuffled_tweeters.tsv and functions.py to work. It follows the following pattern:

- loads the profiles from shuffled_tweeters.tsv
- breaks them into train and test set
- "cleans" the profiles of punctuation and URLs
- vectorizes the profiles using term frequency-inverse document frequency
- trains the model using a random forest
- runs the test set through the model
- mimics "first run"/frequentist calculation - if dr, doctor, or md appears in profile
- outputs:
  - word counts from tf-idf
  - word importances from the model
  - random forest model in forest/ directory
