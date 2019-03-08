import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def static(train,result):
    print("===============================")
    print("檢查模型與預測資料集")
    print("訓練資料集數量 : [",len(train),"]")
    print("測試資料及數量 : [",len(result),"]")

def accurate(get,zero,result):
    print("===============================")
    print("預測正確次數為 : [",get,"]")
    print("預測全為0次數  : [",zero,"]")
    print("預測失敗次數為 : [",len(result)-get-zero,"]")
    return get/len(result)*100
    

if __name__=="__main__":    
    answer =[]
    data = '/Users/aaron/Desktop/Data/train_out.csv'
    train = pd.read_csv(data)
    print("原檔案總資料數量 : ",train.shape)
    print("刪除第一個column : 'Unnamed: 0' ")
    train.pop('Unnamed: 0')
    print("刪除第二個column : 'CUST_ID' ")
    train.pop('CUST_ID')   
        
#    test = train.sample(n=15000) 不拆分的test抽樣
    
    train , test = train_test_split(train,test_size=0.1)
    
    
    test_ans = test[['BUY_TYPE_a','BUY_TYPE_b','BUY_TYPE_c',
                     'BUY_TYPE_d','BUY_TYPE_e','BUY_TYPE_f',
                     'BUY_TYPE_g']]
    
    test.pop('BUY_TYPE_a')
    test.pop('BUY_TYPE_b')
    test.pop('BUY_TYPE_c')
    test.pop('BUY_TYPE_d')
    test.pop('BUY_TYPE_e')
    test.pop('BUY_TYPE_f')
    test.pop('BUY_TYPE_g')
    
    train_ans = train[['BUY_TYPE_a','BUY_TYPE_b','BUY_TYPE_c',
                     'BUY_TYPE_d','BUY_TYPE_e','BUY_TYPE_f',
                     'BUY_TYPE_g']]
    
    train.pop('BUY_TYPE_a')
    train.pop('BUY_TYPE_b')
    train.pop('BUY_TYPE_c')
    train.pop('BUY_TYPE_d')
    train.pop('BUY_TYPE_e')
    train.pop('BUY_TYPE_f')
    train.pop('BUY_TYPE_g')
    
    
    print("===========整理後的SET==========")
    print("Train_Set : ",train.shape)
    print("Ans___Set : ",train_ans.shape)
    print("Test__Set : ",test.shape)
    print("Ans___Set : ",test_ans.shape)
    
    #dataframe -> numpy.array.value
    train_x = train.values
    train_y = train_ans.values
    
    test_x = test.values
    test_y = test_ans.values
    
    #資料預處理
    min_max_scaler = preprocessing.MinMaxScaler()
    
    print("=======進行Random forest========")
    test_result=[]
    forest = RandomForestClassifier(max_depth=35,n_estimators=40)
    forest.fit(train_x,train_y)
    
    #檢查特徵的重要性
    importances = forest.feature_importances_
    indices = np.argsort(importances)
    plt.figure(1)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks()
    plt.xlabel('Relative Importance')

    test_result = forest.predict(test_x)            
    
    print("========整理預測出的答案==========")
    for i in range(0,len(test_result)):
        max_ = test_result[i][0]
        index = 0
        for j in range(1,7):
            if(test_result[i][j]>max_):
                max_= test_result[i][j]
                index = j
        for k in range(0,7):
            if k!=index :
                test_result[i][k]=0
            else:
                test_result[i][k]=1
    
    for i in range(0,len(test_result)):
        for j in range(0,7):
            test_result[i][j] = round(test_result[i][j])
            
    test_result = test_result.astype(int)
   
    
    
    zero=0
    get=0
    for i in range(0,len(test_y)):
        tmp=0
        for j in range(0,7):
            if test_y[i][j] == test_result[i][j]:
                tmp+=1
        if tmp==7:
            get+=1
        elif tmp==6:
            zero+=1
    
    
    static(train_y,test_result)
    acc = accurate(get,zero,test_result)
    print("預測準確率 =",acc,"%")

