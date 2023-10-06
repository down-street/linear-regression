import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import SGDRegressor, LinearRegression
from scipy.stats import probplot
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


def apply_transform(transformer,col):
    plt.figure(figsize=(14,4))
    plt.subplot(131)
    sns.distplot(df[col])
    plt.subplot(132)
    sns.boxplot(df[col])
    plt.subplot(133)
    probplot(df[col],rvalue=True,dist='norm',plot=plt)
    plt.suptitle(f"{col} Before Transform")
    plt.show()
    col_tf = transformer.fit_transform(df[[col]])
    col_tf = np.array(col_tf).reshape(col_tf.shape[0])
    plt.figure(figsize=(14,4))
    plt.subplot(131)
    sns.distplot(col_tf)
    plt.subplot(132)
    sns.boxplot(col_tf)
    plt.subplot(133)
    probplot(col_tf,rvalue=True,dist='norm',plot=plt)
    plt.suptitle(f"{col} After Transform")
    plt.show()
    gc.collect()


def train_and_evaluate_model(model):
    model.fit(final_train,y_train)
    y_pred = model.predict(final_test)
    r2 = r2_score(y_test,y_pred)
    rmse = mean_squared_error(y_test,y_pred,squared=False)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    plot_y.append(r2)
    print("R2 Score:",r2)
    print("RMSE:",rmse)
    print("MAPE:",mape)


df = pd.read_csv('concrete_data.csv')
df = df.drop_duplicates()

# EDA过程
# skewed_cols = ['Cement','Blast Furnace Slag','Water','Superplasticizer','Fine Aggregate','Age','Strength','Fly Ash','Coarse Aggregate']
# for col in skewed_cols:
#     apply_transform(PowerTransformer(),col)
# for col in skewed_cols:
#     apply_transform(FunctionTransformer(np.log1p),col)
# for col in skewed_cols:
#     apply_transform(FunctionTransformer(np.sqrt),col)
# for col in skewed_cols:
#     apply_transform(FunctionTransformer(lambda x: x**2),col)

X = df.drop('Strength',axis=1)
y = df['Strength']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True,random_state=101)

final_selected_features = ['Cement','Blast Furnace Slag','Water','Superplasticizer','Fine Aggregate','Age','Fly Ash','Coarse Aggregate']
final_X_train = X_train[final_selected_features]
final_X_test = X_test[final_selected_features]
transformer = ColumnTransformer(transformers=[
    ('power_transformer',PowerTransformer(),['Cement','Superplasticizer','Water','Coarse Aggregate']),
    ('log_transformer',FunctionTransformer(func=np.log1p,inverse_func=np.expm1),['Age']),
    ('sqrt_tansformer',FunctionTransformer(func=np.sqrt),['Blast Furnace Slag','Fly Ash'])
],remainder='passthrough')

final_X_train_tf = transformer.fit_transform(final_X_train)
final_X_train_tf = pd.DataFrame(final_X_train_tf,columns=final_X_train.columns)
final_X_test_tf = transformer.transform(final_X_test)
final_X_test_tf = pd.DataFrame(final_X_test_tf,columns=final_X_test.columns)
scaler = StandardScaler()
features = final_X_train_tf.columns
final_X_train_tf = scaler.fit_transform(final_X_train_tf)
final_X_train_tf = pd.DataFrame(final_X_train_tf,columns=features)
final_X_test_tf = scaler.transform(final_X_test_tf)
final_X_test_tf = pd.DataFrame(final_X_test_tf,columns=features)

# PCA降维
pca = PCA(n_components=0.95) 
final_X_train_pca = pca.fit_transform(final_X_train_tf)
final_X_test_pca = pca.transform(final_X_test_tf)
final_X_train_pca = pd.DataFrame(final_X_train_pca)
final_X_test_pca = pd.DataFrame(final_X_test_pca)

# final_train = final_X_train_pca
# final_test = final_X_test_pca

final_train = final_X_train_tf
final_test = final_X_test_tf

#train_and_evaluate_model(LinearRegression())
nums = np.linspace(0.00005, 0.1, 500)
plot_x = []
plot_y = []
for i in range(500):
    plot_x.append(nums[i])
    train_and_evaluate_model(SGDRegressor(penalty='l1', alpha=nums[i]))

plt.plot(plot_x, plot_y)
plt.xlabel('l1') 
plt.ylabel('r2')
plt.legend()
plt.grid() 
plt.show()

plot_x = []
plot_y = []
for i in range(500):
    plot_x.append(nums[i])
    train_and_evaluate_model(SGDRegressor(penalty='l2', alpha=nums[i]))
plt.plot(plot_x, plot_y)
plt.xlabel('l2') 
plt.ylabel('r2')
plt.legend()
plt.grid() 
plt.show()