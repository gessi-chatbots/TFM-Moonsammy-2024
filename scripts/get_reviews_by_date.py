import argparse
import json
from datetime import datetime
import csv
import os
import random

def save_matrix_to_csv(matrix, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in matrix:
            csvwriter.writerow(row)


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input-file', required=True, help="Review set.")
ap.add_argument('-a', '--app', required=True, help="The app package.")
ap.add_argument('-d', '--dates', required=True, help="The time window expressed by the closed interval [d1, d2].")
ap.add_argument('-n', '--n', required=True, help="Number of reviews to collect in the sample.")
ap.add_argument('-o', '--output', required=True, help="Output folder for files.")

args = vars(ap.parse_args())

source_file = args['input_file']
source_app = args['app']
from_date = datetime.strptime(args['dates'].split(" - ")[0], '%b %d, %Y')
to_date = datetime.strptime(args['dates'].split(" - ")[1], '%b %d, %Y')
folder = args['output']
n = int(args['n'])

with open(source_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

reviews = []
for app in data:
	if app['package_name'] == source_app:
		for review in app['reviews']:
			review_date = datetime.strptime(review['at'], '%b %d, %Y')
			if review_date >= from_date and review_date <= to_date:
				review = [review['review']]
				reviews.append(review)

# Sort reviews
if not os.path.exists(folder):
	os.makedirs(folder)
	
# Shuffle reviews and get the first 50
random.shuffle(reviews)
reviews = reviews[:n]

# Add prompt engineering at the end
reviews.append([])
reviews.append(["Identify and summarize the most significant event raised by this set of reviews extracted from mobile app repositories."])
	
save_matrix_to_csv(reviews, os.path.join(folder, source_app + "-reviews-" + args['dates'] + ".txt"))