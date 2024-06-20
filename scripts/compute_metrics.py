import argparse
import json
import csv
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from textblob import TextBlob

class ReviewAnalyzer:
    def __init__(self, input_file, time_window, date_range, output_folder):
        self.input_file = input_file
        self.time_window = int(time_window)
        self.date_range = date_range
        self.start_date = datetime.strptime(date_range.split(" - ")[0], '%b %d, %Y')
        self.end_date = datetime.strptime(date_range.split(" - ")[1], '%b %d, %Y')
        self.output_folder = output_folder

    def is_date_in_interval(self, date_str, interval_str):
        date_format = "%b %d, %Y"
        date = datetime.strptime(date_str, date_format)
        interval_start, interval_end = interval_str.split(" - ")
            
        start_date = datetime.strptime(interval_start, date_format)
        end_date = datetime.strptime(interval_end, date_format)
            
        return start_date <= date <= end_date

    def generate_date_intervals(self):
        current_date = self.start_date
        dates = []
        while current_date <= self.end_date:
            interval_end_date = current_date + timedelta(days=self.time_window-1)
            if interval_end_date > self.end_date:
                interval_end_date = self.end_date
            dates.append(f"{current_date.strftime('%b %d, %Y')} - {interval_end_date.strftime('%b %d, %Y')}")
            current_date += timedelta(days=self.time_window)
        return dates

    def compute_app_data(self, app, dates, value_func):
        app_data = [0]*(len(dates)+1)
        app_data[0] = app['package_name']
        
        date_sum = [0]*len(dates)
        date_count = [0]*len(dates)
        
        # Init out structure
        for j, date in enumerate(dates):
            for review in app['reviews']:
                if self.is_date_in_interval(review['at'], date):
                    date_count[j] += 1
                    date_sum[j] += value_func(review)
        
        for j, result in enumerate(date_sum):
            if date_count[j] > 0:
                app_data[j+1] = date_sum[j] / date_count[j]
            else:
                app_data[j+1] = 0
        
        return app_data


    def write_to_csv(self, file_name, data):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        with open(os.path.join(self.output_folder, file_name), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)

    def compute_review_polarity(self):
        dates = self.generate_date_intervals()
        
        # Open the JSON file
        with open(self.input_file) as json_file:
            # Load the contents of the file
            data = json.load(json_file)
        
        output_data = [['App'] + dates]
        negative_data = [['App'] + dates]
        neutral_data = [['App'] + dates]
        positive_data = [['App'] + dates]
        negative_percentage_data = [['App'] + dates]
        neutral_percentage_data = [['App'] + dates]
        positive_percentage_data = [['App'] + dates]
        
        print("Computing review polarity. This might take a while...")
        
        for app in data:
            app_data = [app['package_name']]
            negative_app_data = [app['package_name']]
            neutral_app_data = [app['package_name']]
            positive_app_data = [app['package_name']]
            negative_percentage_app_data = [app['package_name']]
            neutral_percentage_app_data = [app['package_name']]
            positive_percentage_app_data = [app['package_name']]
            
            for date in dates:
                interval_reviews = [review for review in app['reviews'] if self.is_date_in_interval(review['at'], date)]
                
                polarities = [TextBlob(review['review']).sentiment.polarity for review in interval_reviews if review['review']]
                average_polarity = sum(polarities) / len(polarities) if polarities else 0
                app_data.append(average_polarity)
                
                negative_count = sum(1 for polarity in polarities if polarity < 0)
                neutral_count = sum(1 for polarity in polarities if polarity == 0)
                positive_count = sum(1 for polarity in polarities if polarity > 0)
                
                total_count = len(polarities)
                negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0
                neutral_percentage = (neutral_count / total_count) * 100 if total_count > 0 else 0
                positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
                
                negative_app_data.append(negative_count)
                neutral_app_data.append(neutral_count)
                positive_app_data.append(positive_count)
                negative_percentage_app_data.append(negative_percentage)
                neutral_percentage_app_data.append(neutral_percentage)
                positive_percentage_app_data.append(positive_percentage)
            
            output_data.append(app_data)
            negative_data.append(negative_app_data)
            neutral_data.append(neutral_app_data)
            positive_data.append(positive_app_data)
            negative_percentage_data.append(negative_percentage_app_data)
            neutral_percentage_data.append(neutral_percentage_app_data)
            positive_percentage_data.append(positive_percentage_app_data)
        
        self.write_to_csv('review_polarity.csv', output_data)
        self.write_to_csv('negative_sentiment_count.csv', negative_data)
        self.write_to_csv('neutral_sentiment_count.csv', neutral_data)
        self.write_to_csv('positive_sentiment_count.csv', positive_data)
        self.write_to_csv('negative_sentiment_percentage.csv', negative_percentage_data)
        self.write_to_csv('neutral_sentiment_percentage.csv', neutral_percentage_data)
        self.write_to_csv('positive_sentiment_percentage.csv', positive_percentage_data)

    def compute_review_length(self):
        dates = self.generate_date_intervals()
        
        with open(self.input_file) as json_file:
            data = json.load(json_file)
        
        output_data = []
        header = ['App']
        for date in dates:
            header.append(date)
        output_data.append(header)
        
        print("Computing review length. This might take a while...")
        
        for i, app in enumerate(data):
            app_data = self.compute_app_data(app, dates, lambda review: len(review['review'].split()) if review['review'] else 0)
            output_data.append(app_data)
        
        self.write_to_csv('review_word_count.csv', output_data)

    def compute_review_count(self):
        dates = self.generate_date_intervals()

        with open(self.input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    
        output_data = [['App'] + dates]
    
        print("Computing review count. This might take a while...")
    
        for app in data:
            app_data = [app['package_name']]
    
            for date in dates:
                interval_reviews = [
                    review for review in app['reviews']
                    if self.is_date_in_interval(review['at'], date)
                ]
                app_data.append(len(interval_reviews))
    
            output_data.append(app_data)
    
        self.write_to_csv('review_count.csv', output_data)

    def compute_review_rating(self):
        dates = self.generate_date_intervals()
        
        # Open the JSON file
        with open(self.input_file) as json_file:
            # Load the contents of the file
            data = json.load(json_file)
        
        output_data = [['App'] + dates]
        negative_count_data = [[''] + dates]
        positive_count_data = [[''] + dates]
        negative_percentage_data = [[''] + dates]
        positive_percentage_data = [[''] + dates]
        
        print("Computing review rating. This might take a while...")
        
        for app in data:
            app_data = [app['package_name']]
            negative_count_app_data = [app['package_name']]
            positive_count_app_data = [app['package_name']]
            negative_percentage_app_data = [app['package_name']]
            positive_percentage_app_data = [app['package_name']]
            
            for date in dates:
                interval_reviews = [review for review in app['reviews'] if self.is_date_in_interval(review['at'], date)]
                
                scores = [review['score'] for review in interval_reviews]
                average_rating = sum(scores) / len(scores) if scores else 0
                app_data.append(average_rating)
                
                negative_count = sum(1 for score in scores if score <= 3)
                positive_count = sum(1 for score in scores if score >= 4)
                total_count = len(scores)
                
                negative_percentage = (negative_count / total_count) * 100 if total_count > 0 else 0
                positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
                
                negative_count_app_data.append(negative_count)
                positive_count_app_data.append(positive_count)
                negative_percentage_app_data.append(negative_percentage)
                positive_percentage_app_data.append(positive_percentage)
            
            output_data.append(app_data)
            negative_count_data.append(negative_count_app_data)
            positive_count_data.append(positive_count_app_data)
            negative_percentage_data.append(negative_percentage_app_data)
            positive_percentage_data.append(positive_percentage_app_data)
        
        self.write_to_csv('review_rating.csv', output_data)
        self.write_to_csv('negative_review_count.csv', negative_count_data)
        self.write_to_csv('positive_review_count.csv', positive_count_data)
        self.write_to_csv('negative_review_percentage.csv', negative_percentage_data)
        self.write_to_csv('positive_review_percentage.csv', positive_percentage_data)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input-file', required=True, help="Review set.")
    ap.add_argument('-w', '--time-window', required=True, help="Time window length.")
    ap.add_argument('-t', '--dates', required=True, help="The time window expressed by the closed interval [d1, d2].")
    ap.add_argument('-o', '--output-folder', required=True, help="Output folder.")
    args = vars(ap.parse_args())

    analyzer = ReviewAnalyzer(args['input_file'], args['time_window'], args['dates'], args['output_folder'])
    analyzer.compute_review_length()
    analyzer.compute_review_count()
    analyzer.compute_review_rating()
    analyzer.compute_review_polarity()