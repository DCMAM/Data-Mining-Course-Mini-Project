from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn

model = pickle.load(open('C:\-- David --\Bina Nusantara\Semester 5\Database Technology\Data Mining\LF01\Final-project\mini_project\model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    counter = 0
    data1 = float(request.form['val1'])
    data2 = float(request.form['val2'])
    data3 = float(request.form['val3'])
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)

