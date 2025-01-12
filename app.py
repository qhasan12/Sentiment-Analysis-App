from flask import Flask, render_template, request
import pickle
import nltk

app = Flask(__name__)

# Load models and vectorizers
with open('SVM_bow.pkl', 'rb') as file:
    lr_bow = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

# Preprocessing function
def preprocess_review(text):
    import re
    from bs4 import BeautifulSoup
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    nltk.download('stopwords')
    stopword_list = set(stopwords.words('english'))
    ps = PorterStemmer()

    def remove_html(text):
        return BeautifulSoup(text, "html.parser").get_text()

    def remove_specialcharacters(text):
        return re.sub(r'[^a-zA-Z\s]', '', text)

    def remove_stopwords(text):
        tokens = text.split()
        tokens = [word.lower() for word in tokens if word.lower() not in stopword_list]
        return ' '.join(tokens)

    def stem_text(text):
        tokens = text.split()
        tokens = [ps.stem(word) for word in tokens]
        return ' '.join(tokens)

    text = remove_html(text)
    text = remove_specialcharacters(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    user_input = None
    if request.method == 'POST':
        if "text" in request.form:
            # Process direct text input
            user_input = request.form["text"]
            preprocessed_text = preprocess_review(user_input)
            bow_features = cv.transform([preprocessed_text])
            prediction = lr_bow.predict(bow_features)
            sentiment = prediction[0].capitalize()  # Capitalize to match sentiment labels
        
        if "file" in request.files:
            # Process file upload
            file = request.files['file']
            if file:
                text = file.read().decode('utf-8')
                preprocessed_text = preprocess_review(text)
                bow_features = cv.transform([preprocessed_text])
                prediction = lr_bow.predict(bow_features)
                sentiment = prediction[0].capitalize()  # Capitalize to match sentiment labels
                user_input = "Uploaded File"

    return render_template('index.html', sentiment=sentiment, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
