import pandas as pd
from numpy import nan as NA


if __name__=="__main__":
    
    data = '/Users/aaron/Desktop/Data/Submmit_Sample_testing_Set.csv'
    test_submit = pd.read_csv(data)
    test_submit.pop('BUY_TYPE')
    
    data1 = '/Users/aaron/Desktop/Data/test_cust_x_info.csv'
    test_cust = pd.read_csv(data1)
    print(test_cust.shape)
    print(test_cust.describe())
    test_cust = test_cust.sort_values(by = "CUST_ID")

    
    data2 = '/Users/aaron/Desktop/Data/test_buy_x_info.csv'
    test_buy = pd.read_csv(data2) 
    print(test_buy.shape) 
    print(test_buy.describe())
    test_buy = test_buy.sort_values(by = "CUST_ID")
    
    data3 = '/Users/aaron/Desktop/Data/test_tpy_x_info.csv'
    test_tpy = pd.read_csv(data3)
    print(test_tpy.shape)
    print(test_tpy.describe())
    test_tpy = test_tpy.sort_values(by = "CUST_ID")

    
    test_set = pd.merge(test_submit , test_cust , on = "CUST_ID")
    test_set = pd.merge(test_set, test_buy , on = "CUST_ID")
    test_set = pd.merge(test_set , test_tpy , on = "CUST_ID")
    print("印出Merge完並經過排序的train_set")
    print(test_set.shape)
    print(test_set)
    
    print("檢查各個column非缺失值數量")
    print(test_set.count(axis=0)) #檢查各col非缺失值數量
    print("檢查各個column缺失值數量")
    print(test_set.shape[0] - test_set.count(axis=0) )
    
    print("==============================")
    print("刪除缺失值太多的column")
    test_set.pop('BUY_YEAR')
    
    print("==============================")
    print("刪除columns後再檢查缺失值數量")
    print(test_set.shape[0] - test_set.count(axis=0) )
    print("剩下的row數量：",test_set.shape)
            
    #針對非數值之變量進行word to vector
    print(test_set.dtypes) #查看資料格式
    print(test_set['BEHAVIOR_1'].unique())
    
    for i in range(1,4):
        complete_name = 'BEHAVIOR_'
        complete_name += str(i)
        size_mapping = {
           'a': 1,
           'b': 2,
           'c': 3}
        test_set[complete_name] = test_set[complete_name].map(size_mapping)
    
    for i in range(1,5):
        complete_name = 'STATUS'
        complete_name += str(i)
        size_mapping = {
           'a': 1,
           'b': 2,
           NA: 0}
        test_set[complete_name] = test_set[complete_name].map(size_mapping)
        
    size_mapping = {
           'a': 1,
           'b': 2,
           NA: 0}
    test_set['IS_NEWSLETTER'] = test_set['IS_NEWSLETTER'].map(size_mapping)
    
    size_mapping = {
           'a': 1,
           'b': 2,
           NA: 0}
    test_set['CHARGE_WAY'] = test_set['CHARGE_WAY'].map(size_mapping)
        
    for i in range(1,11):
        complete_name = 'INTEREST'
        complete_name += str(i)
        size_mapping = {
           'a': 1,
           'b': 2,
           NA: 0}
        test_set[complete_name] = test_set[complete_name].map(size_mapping)
    
    size_mapping = {
           'a': 1,
           'b': 2,
           'c': 3,
           'd': 4}
    test_set['EDUCATION'] = test_set['EDUCATION'].map(size_mapping)
    
    size_mapping = {
           'a': 0,
           'b': 1}
    test_set['IS_EMAIL'] = test_set['IS_EMAIL'].map(size_mapping)
    
    size_mapping = {
           'a': 0,
           'b': 1}
    test_set['IS_PHONE'] = test_set['IS_PHONE'].map(size_mapping)
    
    size_mapping = {
           'a': 0,
           'b': 1}
    test_set['IS_APP'] = test_set['IS_APP'].map(size_mapping)
    
    size_mapping = {
           'a': 0,
           'b': 1}
    test_set['IS_SPECIALMEMBER'] = test_set['IS_SPECIALMEMBER'].map(size_mapping)
    
    size_mapping = {
           'A': 0,
           'B': 1}
    test_set['PARENTS_DEAD'] = test_set['PARENTS_DEAD'].map(size_mapping)
    
    size_mapping = {
           'A': 0,
           'B': 1}
    test_set['REAL_ESTATE_HAVE'] = test_set['REAL_ESTATE_HAVE'].map(size_mapping)
    
    size_mapping = {
           'A': 0,
           'B': 1}
    test_set['IS_MAJOR_INCOME'] = test_set['IS_MAJOR_INCOME'].map(size_mapping)
    
#    AGE更改
    
    size_mapping = {
           'a': 0,
           'b': 1}
    test_set['SEX'] = test_set['SEX'].map(size_mapping)
    

    
    size_mapping = {
           'a': 0,
           'b': 1,
           'c': 2,
           'd': 3,
           'e': 4,
           'f': 5,
           'g': 6}
    test_set['MARRIAGE'] = test_set['MARRIAGE'].map(size_mapping)
    
    
    for i in range(1,8):
        complete_name = 'BUY_TPY'
        complete_name += str(i)
        complete_name += '_NUM_CLASS'        
        size_mapping = {
               'A': 0,
               'B': 1,
               'C': 2,
               'D': 3,
               'E': 4,
               'F': 5,
               'G': 6}
        test_set[complete_name] = test_set[complete_name].map(size_mapping)
    
    # AGE \ OCCUPATION \ CITY_CODE
    test_set = pd.get_dummies(data=test_set, columns=['AGE'])
    test_set = pd.get_dummies(data=test_set, columns=['CITY_CODE'])
    
#    test_set = pd.get_dummies(data=test_set, columns=['OCCUPATION'])
    occ= ['a37', 'a41', 'b28', 'c32', 'c37', 'c41', 'd10', 'd12', 
          'd14', 'd16', 'd18', 'd22', 'd24', 'd29', 'd3', 'd32', 
          'd37', 'd38', 'd40', 'd41', 'd42', 'd46', 'd47', 'd5', 
          'd7', 'e37', 'e41', 'f12', 'f14', 'f29', 'f32', 'f37', 
          'f40', 'f41', 'f42', 'f46', 'g28', 'h32', 'h37', 'h41', 
          'i37', 'i45', 'j37', 'j41', 'k12', 'k14', 'k29', 'k32', 
          'k37', 'k41', 'l12', 'l32', 'l37', 'l41', 'm28','m37', 'n12', 
          'n14', 'n29', 'n32', 'n37', 'n41', 'o28', 'p11', 'p12',
          'p15', 'p18', 'p19', 'p21', 'p22', 'p24', 'p29', 'p32',
          'p37', 'p38', 'p4', 'p42', 'p43', 'p47', 'p48', 'p49', 
          'p50', 'p6', 'p9', 'q37', 'q41', 'r12', 'r32', 'r37',
          'r41', 's32', 's37', 's41', 't28', 'u37', 'u41', 'v12',
          'v29', 'v41']
    
    for i in occ:
        test_set["OCCUPATION_"+i] = 0
        
        
    for j in range(0,10000):
        for k in range(len(occ)) :
            if test_set['OCCUPATION'][j] == occ[k]:
                tmp = 'OCCUPATION_'+occ[k]
#                print("j :",j,"occ =",tmp," HIT!")
                test_set[tmp][j] = 1
            
    test_set.pop("OCCUPATION")
    
        
    print("==============================")
    print("整理有缺失值的row")
    test_set = test_set.fillna(test_set.mean())
    print("==============================")
    print("檢查處理完的資料維度 : ",test_set.shape)
    print("檢查處理完的資料格式")
    print(test_set.dtypes) #查看資料格式
    print("寫出為新的檔案")
#    test_set.to_csv('/Users/aaron/Desktop/Data/test_out.csv')
    
    
    print(list(test_set.columns.values)) #顯示columns name

    
