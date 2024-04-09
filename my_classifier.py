import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
def reaf_file(file_name):
    with open(file_name,'r') as file:
        data=[]
        lines=file.readlines()
        for line in lines:
            line=line.strip()
            data.append(line)
        return data
def read_file_and_missing_value_imputation(file_name):
    with open(file_name,'r') as file:
        data=[]
        lines=file.readlines()
        for line in lines:
            tmp_dict={'0':"Unknown",'1':'Unknown','2':'Basic','3':'Unknown','4':'Unknown'}
            line=line.strip()[1:-1].split(",")
            for attr in line:
                attr=attr.split(" ")
                tmp_dict[attr[0]]=attr[1][:]
            data.append(list(tmp_dict.values()))
        return data


def data_preprocessing(df,label_mapping):
    ## Marital Status One Hot Encoding
    df=pd.get_dummies(df,columns=['Marital_Status'],prefix='Marital_Status')
    df['Marital_Status_M']=df['Marital_Status_M'].astype(int)
    df['Marital_Status_Unknown']=df['Marital_Status_Unknown'].astype(int)



    attr_list=['Children','Age','Income']
    for attr in attr_list:
        ## missing value imputation
        df[attr]=df[attr].replace('Unknown',np.nan)
        df[attr]=df[attr].astype(float)

        missing_count = df[attr].isnull().sum()
        if missing_count > 0:
            non_missing_values = df[attr].dropna()
            missing_distribution = non_missing_values.value_counts(normalize=True)
            missing_values = np.random.choice(missing_distribution.index,
                                            size=missing_count,
                                            p=missing_distribution.values)
            df.loc[df[attr].isnull(), attr] = missing_values
        df[attr]=df[attr].astype(int)


    ## Normalization
    scaler = MinMaxScaler()
    df[['Children', 'Age', 'Income']] = scaler.fit_transform(df[['Children', 'Age', 'Income']])

    df['Membership_Level'] = df['Membership_Level'].map(label_mapping).astype(int)
    

    return df

class MyKNN:
    def __init__(self,k=5):
        self.k=k

    def fit(self,X_train,y_train):
        self.X_train=X_train.values
        self.y_tain=y_train.values

    def predict(self,X_test):
        X_test=X_test.values
        preds=[]
        for x in X_test:
            distances=np.sqrt(np.sum((self.X_train-x)**2,axis=1))
            nearest_idxs=np.argsort(distances)[:self.k]
            nearest_labels=self.y_tain[nearest_idxs]
            pred=np.argmax(np.bincount(nearest_labels))
            preds.append(pred)
        return np.array(preds)
    
def model_score(y_ans,y_pred):
    accuracy = accuracy_score(y_ans, y_pred)
    precision = precision_score(y_ans, y_pred,average='macro')
    recall = recall_score(y_ans, y_pred,average='macro')
    f1 = f1_score(y_ans, y_pred,average='macro')
    conf_matrix = confusion_matrix(y_ans, y_pred)
    return accuracy,precision,recall,f1,conf_matrix
if __name__=="__main__":

    train_data=read_file_and_missing_value_imputation('training.txt')
    test_data=read_file_and_missing_value_imputation('test.txt')
    columns = ['Marital_Status', 'Children', 'Membership_Level', 'Age', 'Income']
    label_mapping={'Basic':0,'Normal':'1','Silver':2,'Gold':3}
    reverse_label_mapping={0:'Basic',1:'Normal',2:'Silver',3:'Gold'}
    df_train=pd.DataFrame(train_data,columns=columns)
    df_test=pd.DataFrame(test_data,columns=columns)
    df_train=data_preprocessing(df_train,label_mapping)
    df_test=data_preprocessing(df_test,label_mapping)
    # print(df_train.dtypes)
    # print(df_test.dtypes)
    my_knn_classifier=MyKNN(k=9)
    my_knn_classifier.fit(df_train[['Children','Age','Income','Marital_Status_M','Marital_Status_Unknown']],df_train['Membership_Level'])
    predictions=my_knn_classifier.predict(df_test[['Children','Age','Income','Marital_Status_M','Marital_Status_Unknown']])
    acc,precision,recall,f1,conf_matrix=model_score(np.array(df_test['Membership_Level']),np.array(predictions))

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)


    orginal_test_data=reaf_file('test.txt')
    # print(orginal_test_data)
    with open('output.csv','w',newline='') as csvfile:
        columns = ['test_data','predictions']
        writer=csv.writer(csvfile)
        writer.writerow(columns)
        for idx,row in enumerate(orginal_test_data):
            writer.writerow([row,reverse_label_mapping[predictions[idx]]])
