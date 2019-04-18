import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import math
import re
from textblob import TextBlob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import *
from sklearn.ensemble import RandomForestClassifier
import smote_variants as sv

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")

def clean_text(text):
  p=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
  return p.lower()

# Using sentiment analysis(polarity) to score the review text and to categorize the review title. 

def layer1_preprocess(df):
  main_df=df
  main_df=main_df.dropna(subset = ['Review Text'])
  main_df['Review Title']= main_df['Review Title'].fillna('pranab')
  main_df['senti']=0.0
  main_df['senti_2']='wow'
  for i in range(len(main_df)):
    analysis = TextBlob(clean_text(main_df['Review Text'].iloc[i]))
    nrr=analysis.sentiment.polarity
    main_df['senti'].iloc[i]=nrr
    if main_df['Review Title'].iloc[i]=='pranab':
      main_df['senti_2'].iloc[i]='empty3'
    else:
      analysis = TextBlob(clean_text(str(main_df['Review Title'].iloc[i])))
      nrr=analysis.sentiment.polarity
      if nrr>0.0:
        main_df['senti_2'].iloc[i]='happy'
      elif nrr<0.0:
        main_df['senti_2'].iloc[i]='sad'
      else:
        main_df['senti_2'].iloc[i]='neutral'   
  return main_df
    
# This Process will take some minutes..
p1=layer1_preprocess(df_train)
# This Process will take some minutes..
p2=layer1_preprocess(df_test)

p1=p1.drop(columns={'Review Text','Review Title'})
p2=p2.drop(columns={'Review Text','Review Title'})

# Managing the NaN
p1['App Version Code']= p1['App Version Code'].fillna('empty1')
p1['App Version Name']= p1['App Version Name'].fillna('empty2')
p2['App Version Code']= p2['App Version Code'].fillna('empty1')
p2['App Version Name']= p2['App Version Name'].fillna('empty2')

#join two p1 and p2
p2['Star Rating']=0
result = p1.append([p2],sort=True)
#converting the data types
result['App Version Code']=result['App Version Code'].astype(str)
result['App Version Name']=result['App Version Name'].astype(str)
# One hot encoding into columns using dummy variables.
p1_copy=result
dummy1=pd.get_dummies(result['App Version Code'])
dummy2=pd.get_dummies(result['App Version Name'])
dummy3=pd.get_dummies(result['senti_2'])
df_1=pd.concat([p1_copy, dummy1,dummy2,dummy3], axis=1)
df_1=df_1.drop(columns={'App Version Code','App Version Name','senti_2'})
df_1.head()
# Separating the Data
df_f1=df_1[df_1['Star Rating']==0]
df_f2=df_1[df_1['Star Rating']!=0]
# Spliting the Data
x = df_f2.drop(columns={'id','Star Rating'})
y = df_f2['Star Rating']
# Creating the Synthetic Data
oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())
X_samp, y_samp= oversampler.sample(x,y)

# Dividing the Dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_samp,y_samp, random_state=0, test_size=0.25)


# Fitting the model for training
model=RandomForestClassifier(criterion= 'gini',min_samples_split=15,n_estimators=50)
model.fit(X_train,y_train)
model.score(X_train,y_train)
#Predict Output
predicted= model.predict(X_test)


# Output a pickle file for the model
#joblib.dump(model, 'saved_model1.pkl') 

x_test=df_f1.drop(columns={'id','Star Rating'})
predicted2= model.predict(x_test)
final=df_f1.drop(columns={'Star Rating'})
final_dict={'id':df_f1['id'],'Star Rating':predicted2}
final_df=pd.DataFrame(final_dict)
print("saving the file..")
final_df.to_csv("sample_submission.csv",index=False)
print("Process Completed..")
