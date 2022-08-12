from copyreg import pickle
from flask import Flask,render_template
from requests import request
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

#data
irisdata=load_iris()
#model
model =KNeighborsClassifier(n_neighbors=4)


xtrain,xtest,ytrain,ytest=train_test_split(irisdata.data,irisdata.target)

model.fit(xtrain,ytrain)
pickle.dump(model, open("model.pkl", 'wb'))
model=pickle.load(open('model.pkl','rb'))


#not used to see on validation data
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))
result3= loaded_model.score(xtest, ytest)
print(result3)