import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix



# custom transformers

class DropColumns(BaseEstimator, TransformerMixin):
    '''
    Drop columns from DataFrame
    Inherit from the sklearn.base classes (BaseEstimator, TransformerMixi) to make this compatible with scikit-learn’s Pipelines
    '''
    
    # initializer 
    def __init__(self, columns):
        # list of columns we derived that needs to be dropped
        self.columns = columns
        
        
    def fit(self, X, y=None):    #, columns
        # self.columns = columns
        return self
    
    def transform(self, X, y=None):
        # return the dataframe with dropped features
        df_cols = list(X.columns)
        for col in self.columns:
            if col in df_cols:
                X.drop(col, axis=1, inplace=True)
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class str_to_num(BaseEstimator, TransformerMixin):
    '''
    Convert DataFrame columns from Object Dtype to numeric Dtype while performing general cleaning for this dataset
    Inherit from the sklearn.base classes (BaseEstimator, TransformerMixi) to make this compatible with scikit-learn’s Pipelines
    '''
    
    def __init__(self, columns):
        self.columns = columns
    
    def custom_feature_handling(self, X):
        '''
        Convert 'Credit_History_Age' feature from Object Dtype to float; regex then iterate over series to capture from string
        fill empty data pointss in 'Num_Credit_Inquiries' column
        '''
        
        self.temp = np.empty([X.shape[0], 1], dtype=float)
        X['Credit_History_Age'] = X['Credit_History_Age'].str.extract(r'(^[1-9][0-9].* \d|[1-9].* \d)')
        for idx, val in X['Credit_History_Age'].items():
            if type(val)==float:
                self.temp[idx] = np.NaN
                continue
            value = float(int(val[:2])*12) + int(val[-1])
            self.temp[idx] = value
        # could run into problem in the below line: X[col] = numpy array
        X['Credit_History_Age'] = self.temp
        X['Credit_History_Age'].fillna(0, inplace=True)
        
        #fill 'Num_Credit_Inquiries' NaN points with back fill method
        X['Num_Credit_Inquiries'].fillna(method='bfill', inplace=True)        
        return X
    
    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X, y=None):
        for column in self.columns:
            # changes 'None' and 'nan' (str) type to numeric and '_' (invalid) to None in row-wise operation
            X[column] = pd.to_numeric(X[column], errors='coerce')
        X = self.custom_feature_handling(X)
        X = X.fillna(0)
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class categorical_encoding(BaseEstimator, TransformerMixin):
    '''
    Perform Categorical encoding on Ordinal and Nominal Categorical features accordingly
    Inherit from the sklearn.base classes (BaseEstimator, TransformerMixi) to make this compatible with scikit-learn’s Pipelines
    '''
    
    def __init__(self):
        self.ordinal_categorical_features = ['Payment_of_Min_Amount', 'Credit_Mix', 'Credit_Score'] 
        self.nominal_categorical_features = ['Occupation', 'Payment_Behaviour']
        self.features_category_dict = {'Credit_Mix': ['_', 'Good', 'Standard', 'Bad'], 'Payment_of_Min_Amount': ['No', 'NM', 'Yes'], 'Credit_Score': ['Good', 'Standard', 'Poor'],
                             'Occupation': ['Scientist', '_______', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'], 
                             'Payment_Behaviour': ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments', 'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments', 'High_spent_Medium_value_payments', '!@9#%8', 'High_spent_Large_value_payments']}
        self.labelencoder = LabelEncoder()
        self.ordinalencoder = OneHotEncoder()
        
    def fit(self, X=None, y=None):
        return self
        
    
    def transform(self, X, y=None):
        # labelencoding
        for col in self.ordinal_categorical_features:
            X[col] = self.labelencoder.fit_transform(X[col])
        # onehotencoding
        for col in self.nominal_categorical_features:
            col_values = self.features_category_dict[col]
            encoded_cols = pd.DataFrame(self.ordinalencoder.fit_transform(X[[col]]).toarray(), columns=col_values)
            X = X.join(encoded_cols)
            X.drop(col, axis=1, inplace=True)
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

## standardscaler transformer for selected features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class scaler(BaseEstimator, TransformerMixin):
    '''
    implement StandardScaler from sklearn.preprocessing module on selected features/columns
    '''
    
    def __init__(self, columns):
        self.columns = columns
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.is_fitted = True
        self.scaler.fit(X[self.columns])
        return self
    
    def transform(self, X, y=None):
        if not self.is_fitted:
            raise Exception("call fit() before transform() on X")
        X[self.columns] = self.scaler.transform(X[self.columns])
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        X[self.columns] = self.scaler.fit_transform(X[self.columns])
        return X


drop_columns = ['ID', 'Customer_ID', 'Name', 'Month', 'SSN', 'Monthly_Inhand_Salary', 'Type_of_Loan']
str_columns = ['Num_of_Delayed_Payment', 'Amount_invested_monthly', 'Monthly_Balance', 'Age', 'Num_of_Loan', 'Outstanding_Debt', 'Changed_Credit_Limit', 'Annual_Income']
scale_cols = ['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
              'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
              'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

# data
X_train = pd.read_csv("C:\\Users\\manik\\Projects\\Credit-Score-Classification\\train.csv")

# pipeline
custom_transformers = [('drop', DropColumns(columns=drop_columns)), ('convert', str_to_num(columns=str_columns)),  
                ('encode', categorical_encoding()), ('StandardScaler', scaler(columns=scale_cols))]     
    
transformers = Pipeline(custom_transformers)

X = transformers.fit_transform(X_train)

#train test split
X, y = X.drop('Credit_Score', axis=1), X['Credit_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Machine Learning

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.svm import SVC
svm = SVC()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

from sklearn.tree import DecisionTreeClassifier
dectree =DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier()

from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier()

from xgboost import XGBClassifier
xgb = XGBClassifier()


# training and evaluation of various Machine Learning ALgorithms
# def predictions(algorithms):
#     evaluation = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Cross Validation Score', 'Confusion Matrix'])
#     for algo in algorithms:
#         algo.fit(X_train, y_train)
#         pred = algo.predict(X_test) 
        # accuracy = round(metrics.accuracy_score(pred, y_test), 4)
        # cross_val = cross_val_score(algo, X, y, cv=5).mean()
        # conf_matrix = confusion_matrix(pred, y_test) 
        # performance = pd.Series([algo, accuracy, cross_val, conf_matrix])
#         pd.concat([evaluation, performance], axis=0)
#     return evaluation

# algorithms = [logreg, svm, knn, dectree, randfor, adaboost, xgb]
# result = predictions(algorithms)
# print(result)



from sklearn.model_selection import GridSearchCV

# GridSearch for each Machine Learning Algorithm's optimal parameters
parameters_dict = dict() ## (key: str), (value: (ML method(), parameters))
parameters_dict['logreg_parameters'] = (LogisticRegression(), {'C': [0.01, 0.03, 0.1, 0.3, 1, 3]})
parameters_dict['svm_parameters'] = (SVC(), {'kernel': ('linear', 'rbf', 'poly', 'sigmoid', 'precomputed'), 'C': [0,0.01,0.5,0.1,1,2,5,10,50,100,500,1000]})
parameters_dict['knn_parameters'] = (KNeighborsClassifier(), {'n_neighbors': np.arange(51, 151, 2)})
parameters_dict['dectree_parameters'] = (DecisionTreeClassifier(), {'criterion':['gini','entropy', 'log_loss']})
parameters_dict['randfor_parameters'] = (RandomForestClassifier(), {'criterion':['gini','entropy', 'log_loss']})
parameters_dict['adaboost_parameters'] = (AdaBoostClassifier(), {'n_estimators': [10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1], 'algorithm': ['SAMME', 'SAMME.R']})
parameters_dict['xgb_parameters'] = (XGBClassifier(), {'nthread':[4], 'learning_rate': [0.01, 0.03, 0.1, 0.3, 1], 'max_depth': [6, 7, 8]})

results = pd.DataFrame(columns=['algorithm', 'best score', 'best parameters'])
for algo, parameters in parameters_dict.values():
    model = GridSearchCV(algo, parameters)
    model.fit(X, y)
    best_score, best_parameters = model.best_score_, model.best_params_
    result = pd.Series([best_score, best_parameters])
    pd.concat(results, result, axis=0)

print(results)