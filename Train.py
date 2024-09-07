import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreparation:
    def __init__(self,df):
        self.df = df

    def title_categorical(self):
        self.df['Title'] = [i.split(".")[0].split(",")[-1].strip() for i in self.df["Name"]]
        self.df['Title'].loc[(self.df['Title'] == 'Miss')& (self.df['SibSp']>0)] = 'Ms'
        self.df['Title'] = self.df['Title'].replace(['Master','Dr','Rev','Mlle','Major','Col','Don','Mme','Lady','Sir','Capt','the Countess','Jonkheer'],'Others')
        self.df = self.df.drop(['Name','Cabin','Ticket'],axis=1)
        self.family_size(self.df)
    
    def family_size(self):
        self.df = self.title_categorical(self.df)
        self.df['family_size'] = self.df['SibSp'] + self.df['Parch']
        self.df = self.df.drop(['SibSp','Parch'],axis=1)
        self.missing_age(self.df)
    
    def missing_age(self):
        self.df = self.family_size(self.df)
        missing_indices = self.df['Age'].isnull()
        random_value = np.random.randint(30,71,size=missing_indices.sum())
        self.df.loc[missing_indices,'Age'] = random_value
        self.df = self.df.loc[(self.df['Age']<70) & (self.df['Age']>=1)]
        self.encode_gender(self.df)
    
    def encode_gender(self):
        self.df = self.missing_age(self.df)
        self.df.loc[self.df['Sex']=='male','Sex'] = 1
        self.df.loc[self.df['Sex']=='female','Sex'] = 0
        self.family_category(self.df)

    def family_category(self):
        self.df = self.encode_gender(self.df)
        self.df['Alone'] = (self.df['family_size']==0).astype(int)
        self.df['Small_family'] = (self.df['family_size']==1).astype(int)
        self.df['Medium_family'] = (self.df['family_size']==2).astype(int)
        self.df['Large_family'] = (self.df['family_size']>=3).astype(int)
        self.df = self.df.drop('family_size',axis=1)
        self.object_to_numeric(self.df)
    
    def object_to_numeric(self):
        self.df = self.family_category(self.df)
        title_data = pd.get_dummies(new_data12['Title'],dtype=float)
        new_data12 = pd.concat([new_data12,title_data],axis=1)
        embarked_data = pd.get_dummies(new_data12['Embarked'],dtype=float)
        new_data12 = pd.concat([new_data12,embarked_data],axis=1)
        new_data12 = new_data12.drop(['Embarked','Title'],axis=1)
        self.prepare_to_model(new_data12)
    
    def prepare_to_model(self):
        self.df = self.object_to_numeric(self.df)
        self.df = self.df.loc[(self.df['Fare']>0)&(self.df['Fare']<=75)]
        x = self.df.drop('Survived',axis=1)
        y = self.df['Survived']
        x_train,y_train= train_test_split(x,y,random_state=0)
        return x_train,y_train
    
    def main(self):
        x_train,y_train = self.title_categorical(self.df)
        