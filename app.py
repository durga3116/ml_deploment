from flask import Flask ,render_template, request 
import joblib
import sklearn


model = joblib.load('model/logistic_regression.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET','POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = model.predict([text])[0]
        sentiment = 'Positive' if sentiment == 1 else 'Negative'
        return render_template('result.html', text=text, sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)