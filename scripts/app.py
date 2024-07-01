from flask import Flask, request, jsonify
import pandas as pd
import os
import joblib 
from preprocess_reviews import preprocess_reviews
from compute_metrics import ReviewAnalyzer

app = Flask(__name__)
model = joblib.load('best_model.pkl')

def process_reviews(input_file, time_window, date_range, output_folder):
    # Preprocess reviews
    preprocess_reviews(input_file, output_folder)

    # Compute metrics
    analyzer = ReviewAnalyzer(os.path.join(output_folder, 'preprocessed_review_set.json'), time_window, date_range, output_folder)
    analyzer.compute_review_length()
    analyzer.compute_review_count()
    analyzer.compute_review_rating()
    analyzer.compute_review_polarity()

    # Load features for prediction
    base_dir_inference = output_folder
    files = [
        "negative_review_count.csv",
        "negative_review_percentage.csv",
        "negative_sentiment_count.csv",
        "negative_sentiment_percentage.csv",
        "neutral_sentiment_count.csv",
        "neutral_sentiment_percentage.csv",
        "positive_review_count.csv",
        "positive_review_percentage.csv",
        "positive_sentiment_count.csv",
        "positive_sentiment_percentage.csv",
        "review_count.csv",
        "review_polarity.csv",
        "review_rating.csv",
        "review_word_count.csv"
    ]
    feature_files = [os.path.join(base_dir_inference, file) for file in files]

    features = pd.DataFrame()
    for file in feature_files:
        df = pd.read_csv(file)
        feature_name = file.split("/")[-1].split(".")[0]
        df.columns = ["App"] + list(df.columns[1:])
        df = df.melt(id_vars=["App"], var_name="Time window", value_name=feature_name)
        if features.empty:
            features = df
        else:
            features = pd.merge(features, df, on=["App", "Time window"], how="outer")

    X_new = features.drop(columns=["App", "Time window"])
    preprocessed_data = X_new

    return preprocessed_data, features

@app.route('/predict', methods=['POST'])
def predict_events():
    data = request.get_json()
    input_file = data.get('input_file')
    time_window = data.get('time_window')
    date_range = data.get('date_range')
    output_folder = data.get('output_folder')

    if not input_file or not time_window or not date_range or not output_folder:
        return jsonify({'error': 'Missing required parameters'}), 400

    preprocessed_data, features = process_reviews(input_file, time_window, date_range, output_folder)
    predictions = model.predict(preprocessed_data)

    output = []
    for index, prediction in enumerate(predictions):
        event = 'yes' if prediction == 1 else 'no'
        app = features.loc[index, 'App']
        time_window = features.loc[index, 'Time window']
        output.append({'app': app, 'time_window': time_window, 'event': event})

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
