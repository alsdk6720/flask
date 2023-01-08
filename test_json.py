# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:45:35 2023

@author: alsdk
"""

# 환경 준비
import pandas as pd
from sklearn import tree, neighbors, svm, linear_model
from sklearn import model_selection
from sklearn import preprocessing
import pickle
import joblib
import json
import sklearn_json as skljson

# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')
# pd.set_option('display.max_columns', None)


# 전처리 (학습을 위해 object형을 벡터형으로 변환시켜준다.)
def prepro(data):    
    
# =============================================================================
#     OneHotEncoder 클래스 사용
# =============================================================================
    
    # OneHotEncoder (넘파이 배열 반환 / 보이지 않는 범주 값 처리)
    ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
    al = ohe.fit_transform(data[['alone']])
    print(ohe.categories_)
    # numpy.array를 Dataframe으로 변환
    print(pd.DataFrame(al, columns=['alone_'+col for col in ohe.categories_[0]]))
    x = pd.concat([data.drop(columns=['alone']),pd.DataFrame(al,columns=['alone_' + col for col in ohe.categories_[0]])], axis=1)
    joblib.dump(ohe, 't0.t')
    
    # 학습할 columns
    list_c = ['spare','time','special','activity','curiosity','with-whom','where','date']
    list_a = ['sp', 'ti', 'sc', 'ac', 'cu', 'wi', 'wh', 'da']
    
    j = 0
    for i in list_a:
        i = ohe.fit_transform(data[[list_c[j]]])
        string = list_c[j]+'_'
        x1 = pd.concat([x.drop(columns=[list_c[j]]), pd.DataFrame(i, columns=[string + col for col in ohe.categories_[0]])], axis=1)
        # print('\nx1: ',x1)
        x = x1
        j+=1
        joblib.dump(ohe, 't'+str(j)+'.t')
    print(x)

    print(x.shape)
    return x


def get_data():
    data = pd.read_csv("../data/recomm.csv")
    # 결측치 확인
    print(data.info())
    # -> 현재 데이터의 값들이 문자열 형태이다 
    print(data.shape)
    

    # 데이터 분리
    x = data.iloc[:,1:]
    y = data['result']
    
    print(x.shape)
    print(y.shape)
    
    # Label의 분포도? 확인
    print(y.value_counts())
    
    # 문자열인 알파벳을 수치화 해줘야 한다(인코딩)
    x = prepro(x)
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=(11), test_size=0.1)

    return x_train, x_test, y_train, y_test

def make_model():
    # model = tree.DecisionTreeClassifier()
    # model = svm.SVC()
    # model = neighbors.KNeighborsClassifier()
    model = linear_model.LogisticRegression()
    return model

def do_learn():
  x_train, x_test, y_train, y_test = get_data()
  model = make_model()
  
  # cv = 교차 검증 횟수 지정
  model_result = model_selection.cross_val_score(model, x_train, y_train, cv=3)
  print("교차 검증 결과 : ", model_result)
  print("교차검증 결과 평균값 : ", model_result.mean())
  
  model.fit(x_train, y_train)
  score = model.score(x_test, y_test)
  print("점수 : ",score)
  pickle.dump(model,open('m1.m','wb'))
  joblib.dump(model, 'saved_model.pkl')
  skljson.to_json(model, 'js_model.json')
  
do_learn()
print('\n\n\n')
  

def load_data(alone, spare, time, special, activity, curiosity, with_whom, where, date):
    
    data = pd.DataFrame({'alone':[alone], 
                         'spare':[spare],
                         'time' : [time],
                         'special' : [special],
                         'activity' : [activity],
                         'curiosity' : [curiosity],
                         'with-whom' : [with_whom],
                         'where' : [where],
                         'date' : [date]})
    ohe0, ohe1, ohe2, ohe3, ohe4, ohe5, ohe6, ohe7, ohe8 = load_encoder()
    ohe = [ohe0, ohe1, ohe2, ohe3, ohe4, ohe5, ohe6, ohe7, ohe8]
   
    # 예측할 columns
    list_c = ['alone','spare','time','special','activity','curiosity','with-whom','where','date']
    list_a = ['al', 'sp', 'ti', 'sc', 'ac', 'cu', 'wi', 'wh', 'da']
    
    
    print(data)
    
    
    j = 0
    for i in list_a:
        i = ohe[j].transform(data[[list_c[j]]])
        string = list_c[j]+'_'
        x = pd.concat([data.drop(columns=[list_c[j]]), pd.DataFrame(i, columns=[string + col for col in ohe[j].categories_[0]])], axis=1)
        # print('\nx1: ',x1)
        data = x
        j+=1
    print(data)
       
    return data

def load_model():
    model = pickle.load(open("m1.m","rb"))
    model = joblib.load('saved_model.pkl')
    model = skljson.from_json('js_model.json')
    return model
def load_encoder():
    encoder1 = joblib.load('t0.t')
    encoder2 = joblib.load('t1.t')
    encoder3 = joblib.load('t2.t')
    encoder4 = joblib.load('t3.t')
    encoder5 = joblib.load('t4.t')
    encoder6 = joblib.load('t5.t')
    encoder7 = joblib.load('t6.t')
    encoder8 = joblib.load('t7.t')
    encoder9 = joblib.load('t8.t')
    
    return encoder1, encoder2, encoder3, encoder4, encoder5, encoder6, encoder7, encoder8, encoder9

def do_predict(model, alone, spare, time, special, activity, curiosity, with_whom, where, date):
    
    x = load_data(alone, spare, time, special, activity, curiosity, with_whom, where, date)
    
    # model이 예측한 값
    y_pre = model.predict(x)
    print(y_pre)
    
    return y_pre

model = load_model()
do_predict(model, 'y','s','i','y','y','y','r','n','s')


