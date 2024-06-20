import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
import os

def preprocess_text(text):
    if not text:
        return ""

    #Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    #Convert to lowercase
    text = text.lower()

    #Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    #Tokenize 
    tokens = word_tokenize(text)

    #Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def preprocess_reviews(input_file, output_folder, min_rating=None):
    with open(input_file) as json_file:
        data = json.load(json_file)

    for app in data:
        app['reviews'] = [
            review for review in app['reviews']
            if (min_rating is None or review['score'] >= min_rating) and 
               (preprocessed_text := preprocess_text(review['review']).strip()) and
               (review.update({'review': preprocessed_text}) or True)
        ]

    output_file_name = 'preprocessed_review_set.json' if min_rating is None else 'preprocessed_review_set_positive.json'
    output_file = os.path.join(output_folder, output_file_name)

    with open(output_file, 'w') as json_file:
        json.dump(data, json_file)

    print(f"Preprocessed data saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', required=True, help="Review set.")
    parser.add_argument('-o', '--output-folder', required=True, help="Output folder.")
    args = parser.parse_args()

    #all reviews
    preprocess_reviews(args.input_file, args.output_folder)
    #positive reviews
    preprocess_reviews(args.input_file, args.output_folder, min_rating=4)