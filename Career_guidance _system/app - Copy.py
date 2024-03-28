from flask import Flask, render_template,request, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer


model = pickle.load(open('svc.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/predict_your_path", methods=['POST', 'GET'])
def predict_your_path():
    if request =='POST':
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
        
        arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, data21]])
        #pred = model.predict(arr)
        return redirect(url_for('Predicted_Path', arr=arr))
    return render_template('predict_your_path.html')


@app.route('/Predicted_Path', methods=['GET', 'POST'])
def Predicted_Path():
    #Label Encoding: Converting To Numeric values
    #Normalizing the data
    # function to Encode the data into numeric
    def Data_Encode_for_input(input_data):
        labelencoder = LabelEncoder()
        for i in range(1,21):
            data[:,i] = labelencoder.fit_transform(data[:,i])
        return data

    data = Data_Encode_for_input(arr)

    data1_input=data[:,:]
    normalized_input_data1 = Normalizer().fit_transform(data1_input)
    
    input_data__ = pd.DataFrame(normalized_input_data1, columns=['Academic percentage in Operating Systems(1%-100%)',
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
    pred = model.predict(input_data__)   
    
    return render_template('Predicted_Path.html', data=pred)
    



@app.route("/Explore_Careers")
def hello1():
    return render_template('Explore_Careers .html')

@app.route("/View_colleges ")
def hello2():
    return render_template('View_colleges .html')

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
