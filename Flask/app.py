from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import datetime as dt
import calendar
import os

app = Flask(__name__)
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/document')
def doc():
    return render_template('Document.html')


@app.route('/predict', methods=['POST'])
def predict():
    store = request.form.get('store')
    dept = request.form.get('dept')
    date = request.form.get('date')
    isHoliday = request.form['isHolidayRadio']
    size = request.form.get('size')
    temp = request.form.get('temp')
    
    d = dt.datetime.strptime(date, '%Y-%m-%d')
    year = d.year
    month = d.month

    month_name = calendar.month_name[month]

    X_test = pd.DataFrame({
        'Store': [store],
        'Dept': [dept],
        'Size': [size],
        'Temperature': [temp],
        'CPI': [212],
        'MarkDown4': [2050],
        'IsHoliday': [isHoliday],
        'Type_B': [0],
        'Type_C': [1],
        'month': [month],
        'year': [year]
    })
    
    y_pred = model.predict(X_test)
    output = round(y_pred[0], 2)

    return render_template('result.html', output=output, store=store, dept=dept, month_name=month_name, year=year)


if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    port = int(os.getenv('VCAP_APP_PORT', '5000'))
    app.run(debug=False, host='0.0.0.0', port=port)
