from flask import Flask,render_template,request,redirect
import pandas as pd
import joblib
import pickle

s = joblib.load('sss.pkl')
m = joblib.load("mo.joblib")
model = pickle.load(open('model.sav','rb')) 



app = Flask(__name__)
@app.route('/')
@app.route('/main')
def main():
	return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
	values = [[i for i in request.form.values()]]
	c=['baseline value', 'accelerations', 'uterine_contractions','severe_decelerations', 'abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability','mean_value_of_long_term_variability', 'histogram_width','histogram_min', 'histogram_mode', 'histogram_mean', 'histogram_median','histogram_variance', 'histogram_tendency']
	df = pd.DataFrame(values,columns=c)
	tes_s=s.transform(df)
	dd = pd.DataFrame(tes_s,columns=df.columns)
	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	print(dd)
	result=model.predict(dd)
	return render_template('index.html',pre='Result is :{}'.format(result))
    
    
    
    
	


if __name__ == "__main__":
	app.debug=True
	app.run('127.0.0.1',port=7000)