from flask import Flask, make_response, request
from pprint import pprint
import json, re
from nltk.corpus import stopwords, wordnet
import nltk
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ngrok.exe http -host-header=rewrite localhost:5000
stop_words = set(stopwords.words("english"))
stop_words.update([
    "don't", "didn't", "doesn't", "can't", "couldn't", "could've", "i'm",
    "i've", "isn't", "there's", "wasn't", "wouldn't", "a", "also", "am"
])


app = Flask(__name__)

# For Sentence Similary  add extra intents here:
unidentified = {

}


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)

    if wn_tag is None:
        return None

    try:
        return wordnet.synsets(word, wn_tag)[0]
    except:
        return None


def insert_to_db(qt, q, a):
	sql = "insert into `medicarebot` (`query`, `addressed`, `query_type`) values ('{0}', '{1}', '{2}')".format(q, a, qt)
	print(sql)
	con = create_engine("mysql+pymysql://ashhar:password@127.0.0.1:3306/cb")
	con.execute(sql)


def get_query_type_by_sentiment(text):
	sentiment_analyzer = SentimentIntensityAnalyzer()
	score = sentiment_analyzer.polarity_scores(text)
	compound_score = score['compound']
	if compound_score < -0.3:
		return "Critical"
	if -0.3 <= compound_score <= 0.9:
		return "Not Critical"
	return "Not Critical"


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = nltk.pos_tag(nltk.word_tokenize(sentence1))
    sentence2 = nltk.pos_tag(nltk.word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score


def lookup(terms, query):
	enhanced = [l.name() for term in terms for syn in wordnet.synsets(term) for l in syn.lemmas()]

	for category in unidentified:
		if set(enhanced).intersection(unidentified[category]["keywords"]):
			mx = -1
			for index, question in enumerate(unidentified[category["questions"]]):
				score = sentence_similarity(terms, question)
				if score > mx:
					mx = score
					answer = unidentified[category]["answers"][index]
			return answer

	query_type = get_query_type_by_sentiment(query)
	insert_to_db(query_type, query, 'N')
	return "Sorry!  I'm a Chatbot for Healthcare.\n I'm still learning."


def clean(text):
  return ' '.join(
  		i for i in re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()
  		if i not in stop_words and len(i) >= 3
  	)


def process(arg):
	query_result = arg.get('queryResult')
	intent_object = query_result.get('intent')

	if intent_object.get("isFallback"):
		clean_query = clean(text=query_result.get('queryText'))
		pos_tags = nltk.pos_tag(nltk.word_tokenize(clean_query))

		key_terms = list()

		for i in pos_tags:
			if i[1].startswith('JJ') or i[1].startswith('NN') or i[1].startswith('VB'):
				key_terms.append(i[0])

		return lookup(key_terms, clean_query)


@app.route('/post', methods=['POST'])
def parser():
	params = request.get_json(silent=True, force=True)
	pprint(params)
	outcome = process(params)
	response = make_response(json.dumps({"fulfillmentText": outcome, "source": "MHSSCOE"}))
	response.headers['Content-Type'] = 'application/json'
	return response


if __name__ == '__main__':
	app.run(debug=True)

# run Flask app
