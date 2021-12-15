import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import tree
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

cars=pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
# Data Cleaning and Preparation
CompanyName = cars['name'].apply(lambda x : x.split(' ')[0])
cars.insert(3,"CompanyName",CompanyName)
cars.drop(['name'],axis=1,inplace=True)
#print(cars.iloc[:,2])
#print(cars.CompanyName.unique())
#print(cars.CompanyName.unique().shape)

#Fixing invalid values
cars.CompanyName=cars.CompanyName.str.lower()
def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)
replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')
#print(cars.CompanyName.unique())
#print(cars.duplicated())
#print(cars.loc[cars.duplicated()])

#visualize the data
""""
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.selling_price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=cars.selling_price)

plt.show()
print(cars.selling_price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))

"""

"""plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = cars.CompanyName.value_counts().plot(kind='bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car company', ylabel='Frequency of company')

plt.subplot(1,3,2)
plt1 = cars.fuel.value_counts().plot(kind='bar')
plt.title('Fuel Type Histogram')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')


plt.show()"""
"""""
df = pd.DataFrame(cars.groupby(['CompanyName'])['selling_price'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Company Name vs Average Price')
plt.show()
"""""

#categorical to numerical
CompanyName_id_nu,CompanyName_id_na = pd.factorize(cars.CompanyName)
cars.insert(1,"CompanyName_id",CompanyName_id_nu)
cars.drop(['CompanyName'],axis=1,inplace=True)
#print(CompanyName_id_nu)

fuel_nu,fuel_na = pd.factorize(cars.fuel)
cars.insert(2,"fuel_id",fuel_nu)
cars.drop(['fuel'],axis=1,inplace=True)


seller_type_nu,seller_type_na = pd.factorize(cars.seller_type)
cars.insert(3,"seller_type_id",seller_type_nu)
cars.drop(['seller_type'],axis=1,inplace=True)

transmission_nu,transmission_na = pd.factorize(cars.transmission)
cars.insert(3,"transmission_id",transmission_nu)
cars.drop(['transmission'],axis=1,inplace=True)

owner_nu,owner_na = pd.factorize(cars.owner)
cars.insert(3,"owner_id",owner_nu)
cars.drop(['owner'],axis=1,inplace=True)



#feature and target
feature_name=['year','CompanyName_id', 'km_driven', 'fuel_id',"seller_type_id", 'transmission_id']
X=cars.loc[:,feature_name]
y=cars.loc[:,'selling_price']

scaler = MinMaxScaler()
X=scaler.fit_transform(X)
#Modeling
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.40,random_state=1)
RandomF=RandomForestRegressor()
RandomF.fit(x_train,y_train)
y_pre=RandomF.predict(x_test)
pred_train_lr=RandomF.predict(x_train)


def build_model(X, y):
    X = sm.add_constant(X)  # Adding the constant
    lm = sm.OLS(y, X).fit()  # fitting the model
    print(lm.summary())  # model summary
    return X


"""def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return (vif)"""
X_train_new = build_model(x_train,y_train)
print(X_train_new)
#print(checkVIF(X_train_new))

#scatter the result
fig = plt.figure()
plt.scatter(y_test,y_pre)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)
plt.show()
print("mean_squared_error(Train): ",np.sqrt(mean_squared_error(y_train,pred_train_lr)))
print("r2_score(Train): ",r2_score(y_train,pred_train_lr))
print("mean_squared_error(Test): ",np.sqrt(mean_squared_error(y_test,y_pre)))
print("r2_score(Test): ",r2_score(y_test,y_pre))