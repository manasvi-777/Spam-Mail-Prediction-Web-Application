import numpy as np
import pandas as pd #for creating dataframes inorder to structure data
from sklearn.model_selection import train_test_split  #to split data int training and test data
from sklearn.feature_extraction.text import TfidfVectorizer  #to convert text to feature vecotrs/numerical values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

#extracting data from csv file to pandas dataframe
raw_mail_data=pd.read_csv('D:\MANASVI\projects\spam mail detection\mail_data.csv')

mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

#labelling spam mail as 0 and ham mail as 1
mail_data.loc[mail_data['Category']=='spam','Category',]=0
mail_data.loc[mail_data['Category']=='ham','Category',]=1

#seperating data as texts and label
X=mail_data['Message']
Y=mail_data['Category']

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=3) #test_size->20% of data for testing data and 80% data goes to training data

#transform text data to feature vectors that can be used as input to the Logistic Regression Model
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True) #min_df=1-> minimum document frequency; sotop_words='english'->common words are filtered out
X_train_features=feature_extraction.fit_transform(X_train)  #fits all the data into TfidVectorizer which is further transformed into feature vectors
X_test_features=feature_extraction.transform(X_test)  #do not fit X_test to TfidVectorizer as we dont want the model to look at the test data

#convert Y_train and Y_test values as integers
Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

model=LogisticRegression()
model.fit(X_train_features, Y_train)

# Make predictions on the test set
Y_pred = model.predict(X_test_features)

# Evaluate the model's accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Save the trained model to a file
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer to a file
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)
