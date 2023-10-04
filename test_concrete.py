import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, gc, joblib
# warnings.filterwarnings('ignore')
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor, Lasso, Ridge, ElasticNet, HuberRegressor, ARDRegression, RANSACRegressor, TweedieRegressor, PoissonRegressor, BayesianRidge, SGDRegressor, GammaRegressor, TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBRFRegressor
# from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
from scipy.stats import probplot
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, f_regression, RFE, SequentialFeatureSelector
from sklearn.compose import ColumnTransformer
# from feature_engine.outliers import Winsorizer
# from sklearn.pipeline import Pipeline

# def apply_transform(transformer,col):
#     plt.figure(figsize=(14,4))
#     plt.subplot(131)
#     sns.distplot(df[col])
#     plt.subplot(132)
#     sns.boxplot(df[col])
#     plt.subplot(133)
#     probplot(df[col],rvalue=True,dist='norm',plot=plt)
#     plt.suptitle(f"{col} Before Transform")
#     plt.show()
#     col_tf = transformer.fit_transform(df[[col]])
#     col_tf = np.array(col_tf).reshape(col_tf.shape[0])
#     plt.figure(figsize=(14,4))
#     plt.subplot(131)
#     sns.distplot(col_tf)
#     plt.subplot(132)
#     sns.boxplot(col_tf)
#     plt.subplot(133)
#     probplot(col_tf,rvalue=True,dist='norm',plot=plt)
#     plt.suptitle(f"{col} After Transform")
#     plt.show()
#     gc.collect()

# def plot_feature_importances(feat_imp_type):
#     feat_imps = xgb.get_booster().get_score(importance_type=feat_imp_type)
#     keys = list(feat_imps.keys())
#     values = list(feat_imps.values())
#     feat_imps_df = pd.DataFrame(data=values, index=keys, columns=["Importance"]).sort_values(by="Importance", ascending=False).reset_index()
#     feat_imps_df.rename({'index': 'Feature'},axis=1,inplace=True)
#     fig = sns.barplot(x='Importance',y='Feature',data=feat_imps_df,orient='horizontal',palette='viridis')
#     plt.title(f"{feat_imp_type.title()} Feature Importance")
#     plt.show(fig)
#     plt.close('all')
#     del fig
#     gc.collect()

def train_and_evaluate_model(model):
    model.fit(final_X_train_tf,y_train)
    y_pred = model.predict(final_X_test_tf)
    r2 = r2_score(y_test,y_pred)
    rmse = mean_squared_error(y_test,y_pred,squared=False)
    mape = mean_absolute_percentage_error(y_test,y_pred)
    print("R2 Score:",r2)
    print("RMSE:",rmse)
    print("MAPE:",mape)
    gc.collect()


df = pd.read_csv('concrete_data.csv')
df = df.drop_duplicates()

# skewed_cols = ['Cement','Blast Furnace Slag','Water','Superplasticizer','Fine Aggregate','Age','Strength','Fly Ash','Coarse Aggregate']
# # for col in skewed_cols:
# #     apply_transform(PowerTransformer(),col)

X = df.drop('Strength',axis=1)
y = df['Strength']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True,random_state=101)

final_selected_features = ['Cement','Blast Furnace Slag','Water','Superplasticizer','Fine Aggregate','Age','Fly Ash','Coarse Aggregate']
final_X_train = X_train[final_selected_features]
final_X_test = X_test[final_selected_features]
transformer = ColumnTransformer(transformers=[
    ('power_transformer',PowerTransformer(),['Cement','Superplasticizer','Water','Coarse Aggregate']),
    ('log_transformer',FunctionTransformer(func=np.log1p,inverse_func=np.expm1),['Age']),
    ('sqrt_tansformer',FunctionTransformer(func=np.sqrt),['Blast Furnace Slag','Fly Ash']),
    ('x2_transformer',FunctionTransformer(func=lambda x: x**2),['Fine Aggregate'])
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

train_and_evaluate_model(LinearRegression())
