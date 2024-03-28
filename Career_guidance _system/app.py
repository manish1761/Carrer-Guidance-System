from flask import Flask, render_template,request, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer

Y = pd.read_csv('Yfile.csv')


model = pickle.load(open('svc.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')



@app.route("/predict_your_path", methods=['POST', 'GET'])
def predict_your_path():
    if request.method == 'POST':
        data1 = request.form['name1']
        data2 = request.form['name2']
        data3 = request.form['name3']
        data4 = request.form['name4']
        data5 = request.form['name5']
        data6 = request.form['name6']
        data7 = request.form['name7']
        data8 = request.form['name8']
        data9 = request.form['name9']
        data10 = request.form['name10']
        data11 = request.form['name11']
        data12 = request.form['name12']
        data13 = request.form['name13']
        data14 = request.form['name14']
        data15 = request.form['name15']
        data16 = request.form['name16']
        data17 = request.form['name17']
        data18 = request.form['name18']
        data19 = request.form['name19']
        data20 = request.form['name20']
        data21 = request.form['name21']
        
        # Convert string elements to integers for label encoding
        arr = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, data21]

        x_new_encoded = []
        for i in range(len(arr)):
            if i < 14 or i == 18:
                try:
                    x_new_encoded.append(int(arr[i]))
                except ValueError:
                    x_new_encoded.append(arr[i])
            else:
                x_new_encoded.append(arr[i])

        # Convert to NumPy array for label encoding
        x_new_array = np.array(x_new_encoded).reshape(1, -1)

        # Label encoding
        label_encoder = LabelEncoder()  
        x_new_encoded = label_encoder.fit_transform(x_new_array.ravel())

        # Reshape back to the original shape
        x_new_encoded = x_new_encoded.reshape(x_new_array.shape)

        # Normalizing the data
        normalized_input_data = Normalizer().fit_transform(x_new_encoded)
                
        input_data__ = pd.DataFrame(normalized_input_data, columns=['Academic percentage in Operating Systems(1%-100%)',
        'percentage in Algorithms(1%-100%)',
        'Percentage in Programming Concepts(1%-100%)',
        'Percentage in Software Engineering(1%-100%)',
        'Percentage in Computer Networks(1%-100%) ',
        'Percentage in Computer Architecture(1%-100%',
        'Percentage in Mathematics(1%-100%)',
        'Percentage in Communication skills(1%-100%)',
        'Hours working per day(0-12)',
        'Logical Reasoning Score(1-10)',
        'No. of hackathons',
        'coding skills rating(1-10)',
        'public speaking Score(1-10)',
        'self-learning capability?(YES/NO)',
        'Which Programming Language/Technology of certifications you did?',
        'reading and writing skills(EXCELLENT/MEDIUM/POOR)',
        'interested career area',
        'interested in games(YES/NO)',
        'Interested Type of Books',
        'Management or Technical(Management/Technical)',
        'worked in teams ever?(YES/NO)'])
        result  = model.predict(input_data__)
        # Assuming `Y` is a DataFrame containing 'Associated Number' and 'Suggested Job Role' columns
        associated_number = result[0]

        if associated_number in Y['Associated Number'].values:
            suggested_role = Y.loc[Y['Associated Number'] == associated_number, 'Suggested Job Role'].values[0]
            return render_template('Predicted_Path.html', data=suggested_role)
        else:
            # Handle the case where the predicted value is not found
            return render_template('Predicted_Path.html', data="Prediction not found")
    return render_template('predict_your_path.html')



@app.route("/Explore_Careers")
def hello1():
    return render_template('Explore_Careers.html')

@app.route("/View_colleges ")
def hello2():
    return render_template('View_colleges.html')

@app.route("/feedback ")
def hello3():
    return render_template('feedback.html')

@app.route("/FAQs")
def hello4():
    return render_template('FAQs.html')

@app.route("/help")
def hello5():
    return render_template('help.html')



if __name__ == '__main__':  
    app.run(debug=True)
