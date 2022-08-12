from copyreg import pickle
from unittest import result
#from crypt import methods
from flask import Flask,render_template
from requests import request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import numpy as np
import pickle
import os
from flask import request
#1
app=Flask(__name__, template_folder='templates')
loaded_model = pickle.load(open('model.pkl', 'rb'))
#function home model on iris
@app.route("/")
def home():
#3rout cllassfiction
   # irisdata=load_iris()

    #model =KNeighborsClassifier(n_neighbors=4)
    #xtrain,xtest,ytrain,ytest=train_test_split(irisdata.data,iris.target)

    #model.fit(xtrain,ytrain)
    #pickle.dumps(model,open("iris.pkl","wb"))
    return render_template("home.html")



#4route prediction
@app.route("/predict",methods=["get","post"])
def predict():
    sepal_length=request.form['sepal_length']
    sepal_width=request.form['sepal_width']

    petal_length=request.form['petal_length']
    petal_width=request.form['petal_width']
    
#take input from user form put in arry
    form_arry=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    #model.pickle.load(open("model.pkl","rb"))
    #predict on arry of user input form
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    prediction=loaded_model.predict(np.array(form_arry.astype(int)))
    classes = ["Iris setosa","Iris Veersiclor","Iris virginica"  ]
    result = classes[int(prediction)]
    # if  int(prediction)== 0:
    #     result="Iris setosa"
    # elif int(prediction) == 1:
    #     result="Iris Veersiclor"  
    # else: 
    #     result="Iris virginica"    


    
    return render_template("result.html",result=result)




#2
if __name__ =="__main__":
    app.run(debug=True)