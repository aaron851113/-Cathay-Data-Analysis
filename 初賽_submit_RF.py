import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def static(train,result):
    print("檢查模型與預測資料集")
    print("訓練資料集數量 : [",len(train),"]")
    print("測試資料及數量 : [",len(result),"]")
   

if __name__=="__main__":    
    answer =[]
    data = '/Users/aaron/Desktop/Data/train_out.csv'
    train = pd.read_csv(data)
    print("訓練檔案總資料數量 : ",train.shape)
    print("刪除第一個column : 'Unnamed: 0' ")
    train.pop('Unnamed: 0')
    print("刪除第二個column : 'CUST_ID' ")
    train.pop('CUST_ID')   
    
    data = '/Users/aaron/Desktop/Data/test_out.csv'
    test = pd.read_csv(data)
    print("預測檔案總資料數量 : ",test.shape)
    print("刪除第一個column : 'Unnamed: 0' ")
    test.pop('Unnamed: 0')
    print("刪除第二個column : 'CUST_ID' ")
    test.pop('CUST_ID')
        
#    test = train.sample(n=15000) 不拆分的test抽樣
    

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
    
    #dataframe -> numpy.array.value
    train_x = train.values
    train_y = train_ans.values
    test_x = test.values


    print("=======進行Random forest========")
    test_result=[]
    forest = RandomForestClassifier(max_depth=35,n_estimators=42)
    forest.fit(train_x,train_y)
    
    
    test_result = forest.predict(test_x)            
    
    print("========整理預測出的答案==========")
    
    for i in range(100):
        sum=0
        for j in range(7):
            sum+=test_result[i][j]
        print("[",i," ]",sum)
    
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
    
    static(train_y,test_result)
    
    
    print("輸出預測答案至[Submit_一南尬三北]")
    data = '/Users/aaron/Desktop/Data/Submmit_Sample_testing_Set.csv'
    submit = pd.read_csv(data)
    submit.pop('BUY_TYPE')
    submit['BUY_TYPE'] = '0'
    for i in range(0,100):
        for j in range(0,7):
            if test_result[i][j]==1:
                if j==0:
                    submit['BUY_TYPE'][i]='a'
                elif j==1:
                    submit['BUY_TYPE'][i]='b'
                elif j==2:
                    submit['BUY_TYPE'][i]='c'
                elif j==3:
                    submit['BUY_TYPE'][i]='d'
                elif j==4:
                    submit['BUY_TYPE'][i]='e'
                elif j==5:
                    submit['BUY_TYPE'][i]='f'
                elif j==6:
                    submit['BUY_TYPE'][i]='g'
        print(i,'= [',submit['BUY_TYPE'][i],']')
    
    submit.to_csv('/Users/aaron/Desktop/Data/Submmit_RF.csv')



