# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import language_tool_python


#filename = 'bot-model.pkl'
rfcclassifier = joblib.load(open('RFC-20%.pkl', 'rb'))
DTClassifier = joblib.load(open('DT-30%.pkl','rb'))
LRClassifier = joblib.load(open('LR-30%.pkl','rb'))
SVCClassifier = joblib.load(open('SVC-20%.pkl','rb'))


cv = pickle.load(open('cv-transform.pkl','rb'))
#cv = pickle.load(open('count.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	#return render_template('home.html')
	return render_template('bot.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		messages = request.form['message']
		friends = request.form['friends']
		followers = request.form['followers']
		statuses = request.form['status']

		status = int([statuses][0])
		friend = int([friends][0])
		follower = int([followers][0])
			
		#print(message)
		#dt = request.form['dt']

		tweet = [messages]
		# print(status)
		# print(friend)
		# print(follower)
		# print(tweet)


		#vect = cv.transform(data).toarray()
		
		vect = np.concatenate((np.array([follower,friend,status]).reshape(1,-1),cv.transform(tweet).toarray()),axis=1)

		
		# Predicting bot or not using four algorithms and find the best prediction
		my_prediction = rfcclassifier.predict(vect)
		my_prediction2 = DTClassifier.predict(vect)
		my_prediction3 = LRClassifier.predict(vect)
		my_prediction4 = SVCClassifier.predict(vect)
		count=0
		predict = [my_prediction,my_prediction2,my_prediction3,my_prediction4]
		for i in predict:
			if i[0]==1:
				count+=1
		# Finding the most similar prediction using the count variable
		if count>=3:
			phase1 = 1
		else:
			phase1 = 0
		
		#using language tool we are checking the tweet is grammatically correct or not.
		tool = language_tool_python.LanguageTool("en-US")
		matches = tool.check(tweet)
		mistake = len(matches)
		correct_text = tool.correct(tweet)
		if mistake>0:
			phase2 = 0
		else:
			phase2 = 1

		phase3 = 0
		#bot - 1     human - 0
		if(follower >100 and friend>100 and status>100):
			phase3 = 0
		if(follower <50 and friend<50 and status<50):
			phase3 = 0
		if(follower >50 and friend==0 and status==0):
			phase3 = 1
		if(follower ==0 and friend>50 and status==0):
			phase3 = 1
		if(follower ==0 and friend==0 and status>50):
			phase3 = 1

		res = 0
		if(phase1==1 and phase2==1 and phase3==1):
			res=1
		if(phase1==0 and phase2==1 and phase3==1):
			res=1
		if(phase1==1 and phase2==0 and phase3==1):
			res=1
		if(phase1==1 and phase2==1 and phase3==0):
			res=1
		if(phase1==0 and phase2==0 and phase3==1):
			res=0
		if(phase1==0 and phase2==1 and phase3==0):
			res=0
		if(phase1==1 and phase2==0 and phase3==0):
			res=0
		if(phase1==0 and phase2==0 and phase3==0):
			res=0

		# print(phase1)
		# print(phase2)
		# print(phase3)

		return render_template('result.html', prediction=res)

if __name__ == '__main__':
	app.run(debug=True,port=4004)
	