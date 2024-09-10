# %%
import pandas as pd
import numpy as np
data=pd.read_csv('Bengaluru_House_Data.csv')

# %%
print("This is the output of data.head \n",data.head())
print("This is the output of data.shape \n",data.shape)
print("This is the output of data.info \n",data.info())
# %%
print("8"*20)
for column in data.columns:
    print(data[column].value_counts())
    print("*"*20) 

print(data.isna().sum())
data.drop(columns=['area_type','availability','society','balcony'],inplace=True)
print("THIS IS DATA DESCRIBE \n",data.describe())
data.info()
# %%

print(data['location'].value_counts())
# %%
data['location'] = data['location'].fillna('Sarjapur Road')
print(data['size'].value_counts())
# %%
data['size'] = data['size'].fillna('2 BHK')
data['bath'] = data['bath'].fillna(data['bath'].median())
print("THIS IS THE OUTPUT OF DATA.INFO \n",data.info())
# %%
data["bhk"] = data["size"].str.split().str.get(0).astype(int)
 #bedroom / #BHK
print(data[data.bhk>10])

print(data['total_sqft'].unique())
#%%
def convertRange(x):

    temp = x.split('-')
    if len(temp)==2:
        return (float(temp[0])+float(temp[1])/2) 
    try:
        return float(x)
    except:
        return None
    
data['total_sqft']=data['total_sqft'].apply(convertRange) #passing function reference on the column
data.head()


# #price per squarefeet
#%%
data['price_per_sqft'] = data['price']*100000/data['total_sqft']
data['price_per_sqft']
data.describe()
print(data.shape)
# %%
data['location'].value_counts()

# %%
data['location'] = data['location'].apply(lambda x: x.strip() if isinstance(x,str) else x)
location_count = data['location'].value_counts()

print(location_count)
location_count.describe()
# %%
location_count_less_10 = location_count[location_count<=10]
print(location_count_less_10)

data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_10 else x)

data['location'].value_counts()
data.describe()
# %% 
print((data['total_sqft']/data['bhk']).describe())

data=data[((data['total_sqft']/data['bhk']) >= 300)]
data.describe()

print(data.shape)

print((data['total_sqft']/data['bhk']).describe())

data.price_per_sqft.describe()

# %%
def remove_outliers_sqft(df):
    df_output = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        gen_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_output = pd.concat([df_output,gen_df],ignore_index = True)
    return df_output
data = remove_outliers_sqft(data)
data.describe()
# %%
def bhk_outlier_remover(df):  #25:00
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats ={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            print(location,bhk_stats)
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
# %%
data=bhk_outlier_remover(data)
print(data.shape)
# %%
print(data)
# %%
data.drop(columns=['size','price_per_sqft'],inplace=True)
data.head()
# %%
data.to_csv("CLeaned_data.csv")
# %%
X=data.drop(columns=['price'])
y=data['price']
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)
# %%
# Applying Linear Regression
column_trans = make_column_transformer((OneHotEncoder(sparse_output=False),['location']),remainder='passthrough')
scaler = StandardScaler()
lr=LinearRegression()
pipe=make_pipeline(column_trans,scaler,lr)
pipe.fit(X_train,y_train)
y_pred_lr = pipe.predict(X_test)
r2_score(y_test,y_pred_lr)

# %%
# Applying Lasso
lasso = Lasso()
pipe = make_pipeline(column_trans,scaler,lasso)
pipe.fit(X_train,y_train)
y_pred_lasso = pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)
# %%
# Applying Ridge
ridge = Ridge()
pipe = make_pipeline(column_trans,scaler,ridge)
pipe.fit(X_train,y_train)
y_pred_ridge = pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)
# %%
print("No Regularization",r2_score(y_test,y_pred_lr))
print("Lasso:", r2_score(y_test,y_pred_lasso))
print("Ridge:",r2_score(y_test,y_pred_ridge))
# %%
import pickle
with open('RidgeModel.pkl','wb') as file:
    pickle.dump(pipe,file)
# %%
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
pipe = make_pipeline(column_trans, scaler, dt)
pipe.fit(X_train, y_train)
y_pred_dt = pipe.predict(X_test)
print("Decision Tree R²:", r2_score(y_test, y_pred_dt))

# %%
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0)
pipe = make_pipeline(column_trans, scaler, rf)
pipe.fit(X_train, y_train)
y_pred_rf = pipe.predict(X_test)
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

# %%
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100)
pipe = make_pipeline(column_trans, scaler, gbr)
pipe.fit(X_train, y_train)
y_pred_gbr = pipe.predict(X_test)
print("Gradient Boosting R²:", r2_score(y_test, y_pred_gbr))

# %%
import xgboost as xgb

xgboost_model = xgb.XGBRegressor(n_estimators=100)
pipe = make_pipeline(column_trans, scaler, xgboost_model)
pipe.fit(X_train, y_train)
y_pred_xgb = pipe.predict(X_test)
print("XGBoost R²:", r2_score(y_test, y_pred_xgb))

# %%
from sklearn.svm import SVR
# Support Vector Regressor
svr = SVR(kernel='linear')
pipe = make_pipeline(column_trans, scaler, svr)
pipe.fit(X_train, y_train)
y_pred_svr = pipe.predict(X_test)
print("SVR R²:", r2_score(y_test, y_pred_svr))

# %%
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
pipe = make_pipeline(column_trans, scaler, knn)
pipe.fit(X_train, y_train)
y_pred_knn = pipe.predict(X_test)
print("KNN R²:", r2_score(y_test, y_pred_knn))

# %%
from sklearn.linear_model import ElasticNet

elasticnet = ElasticNet()
pipe = make_pipeline(column_trans, scaler, elasticnet)
pipe.fit(X_train, y_train)
y_pred_enet = pipe.predict(X_test)
print("ElasticNet R²:", r2_score(y_test, y_pred_enet))

# %%
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=500)
pipe = make_pipeline(column_trans, scaler, mlp)
pipe.fit(X_train, y_train)
y_pred_mlp = pipe.predict(X_test)
print("MLP R²:", r2_score(y_test, y_pred_mlp))

# %%
# Averaging the predictions of multiple models
y_pred_lr = pipe.predict(X_test)          # Linear Regression predictions
y_pred_ridge = pipe.predict(X_test)       # Ridge predictions
y_pred_rf = pipe.predict(X_test)          # Random Forest predictions
y_pred_xgb = pipe.predict(X_test)         # XGBoost predictions
y_pred_gbr = pipe.predict(X_test)         # Gradient Boosting predictions

# Take the average of the predictions
y_pred_avg = (y_pred_lr + y_pred_ridge + y_pred_rf + y_pred_xgb + y_pred_gbr) / 5

# Evaluate the averaged predictions
avg_r2_score = r2_score(y_test, y_pred_avg)
print("Averaging method R²:", avg_r2_score)

# %%
# %%
# Predictions from each model
y_pred_lr = pipe.predict(X_test)          # Linear Regression predictions
y_pred_ridge = pipe.predict(X_test)       # Ridge predictions
y_pred_rf = pipe.predict(X_test)          # Random Forest predictions
y_pred_xgb = pipe.predict(X_test)         # XGBoost predictions
y_pred_gbr = pipe.predict(X_test)         # Gradient Boosting predictions

# R² scores from previous cells
r2_lr = r2_score(y_test, y_pred_lr)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)
r2_gbr = r2_score(y_test, y_pred_gbr)

# Determine the weights based on R² scores
weights = np.array([r2_lr, r2_ridge, r2_rf, r2_xgb, r2_gbr])
weights = weights / np.sum(weights)  # Normalize the weights

# Compute the weighted average of predictions
y_pred_weighted_avg = (
    weights[0] * y_pred_lr +
    weights[1] * y_pred_ridge +
    weights[2] * y_pred_rf +
    weights[3] * y_pred_xgb +
    weights[4] * y_pred_gbr
)

# Evaluate the weighted average predictions
weighted_avg_r2_score = r2_score(y_test, y_pred_weighted_avg)
print("Weighted Average method R²:", weighted_avg_r2_score)

# %%
