from flask import Flask ,render_template , request ,redirect
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS , cross_origin

app = Flask(__name__)
cors=CORS(app)
model = pickle.load(open('gboost2.pkl','rb'))

@app.route('/' , methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict' , methods = ['POST'])
@cross_origin()
def predict():

    floor_numbers = int(request.form.get('floor_numbers'))
    house_area = int(request.form.get('house_area'))
    house_age = int(request.form.get('house_age'))
    renovated = int(request.form.get('renovated'))
    bathrooms = int(request.form.get('bathrooms'))
    coastline = int(request.form.get('coastline'))

    prediction = model.predict(pd.DataFrame(columns=['ceil' , 'age' , 'renovated' , 'room_bath', 'coast' , 'living_measure15'],
                                            data=np.array([floor_numbers , house_age , renovated , bathrooms, coastline, house_area])
                                            .reshape(1,6)))
    print(prediction)

    return str(np.round(np.exp(prediction[0]),2))


if __name__ == '__main__':
      app.run(debug = True)


