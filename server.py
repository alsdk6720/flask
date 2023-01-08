# -*- coding: utf-8 -*-


# 환경 준비
import pandas as pd
from flask import Flask, request, render_template
import joblib

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
    model = joblib.load('saved_model.pkl')
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
# do_predict(model, 'y','s','i','y','y','y','r','n','s')

# =============================================================================
#  flask
# =============================================================================

webserver = Flask(__name__)
print(__name__)


@webserver.route("/")
def index():
    msg = "Welcome! this is mainPage"
    return msg

@webserver.route("/recomm_My_")
def recomm_My_():
    msg = render_template("testStart.html")
    return msg

# method가 POST인 형식만 받겠다.
@webserver.route("/recomm_My_/ans", methods=["POST"])
def recomm_My_result():
    alone = request.values.get('al')
    spare = request.values.get('sp')
    time = request.values.get('ti')
    special = request.values.get('sc')
    activity = request.values.get('ac')
    curiosity = request.values.get('cu')
    with_whom = request.values.get('wi')
    where = request.values.get('wh')
    date = request.values.get('da')
    
    y_pre = do_predict(model, alone, spare, time, special, activity, curiosity, with_whom, where, date)
    pred=y_pre[0]
    print(pred)
    # if pred == "e":
    #     fin = render_template("result.html")
    # elif pred == "p":
    #     fin = render_template("result1.html")
     
    return pred


if __name__ == '__main__':
    webserver.run(host='0.0.0.0', port=5000, debug=True)