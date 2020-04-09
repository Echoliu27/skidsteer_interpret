import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ml
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# image
from PIL import Image

# interpretation 
import shap
import eli5
from eli5.sklearn import PermutationImportance

# plotly
import plotly.express as px

# DONE:
# [🎉] Sample tabular data
# [🎉] Display global and local interpretation
# [🎉] Add PDP chart
# [🎉] Allow csv upload
# [🎉] auto encode
# [🎉] Filter for misclassification
# [🎉] deploy to heroku
# [🎉] add more ml algos: xgb, lgbm
# [🎉] add confusion matrix
# TODO:
# [ ] fix pred
# [ ] turn the first table into a plot
# [ ] add another demo data
# [ ] add pdp for xgb
# [ ] add distribution plot for individual datapoint
# [ ] add circleCI
# GOOD-TO-HAVE:
# [ ] css formating
# [ ] Add shields.io
# [ ] Allow model upload
# [ ] add other interpretation framework (SHAP etc)
# [ ] add two variable interaction pdp (pending pdpbox maintainer fix)
# [ ] Add other data types: text, image


###############################
## Random Forest Page
###############################
@st.cache
def read_split_data_rf():
	# Read and process tabular data : only 96 rows of data
	df_orig = pd.read_csv('data/final_unscaled.csv')
	df_tabular = pd.read_csv('data/final_tabular.csv')
	df_mturk = pd.read_csv('data/mturk_96.csv')
	df_join = pd.merge(df_tabular, df_mturk, on=['unique_id'], how='right')

	# normalize columns
	def normalize_scaler(df_join, col):
	    std_scaler = preprocessing.StandardScaler()
	    df_join[col] = std_scaler.fit_transform(np.array(df_join[col]).reshape(-1, 1))

	num_col_list = ['body_rust_extent', 'buck_rust_extent','tread_wear','paint_brightness','cabin_blackness','dirt_level']
	for col in num_col_list:
	    normalize_scaler(df_join, col)

	# split into train and test
	y = df_join['winning_bid']
	X = df_join.drop(['winning_bid','model','hours_final_nan','age_at_sale_nan','bucket_x','engine','tires','transmission'],axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)
	
	return X_train, X_test, y_train, y_test, df_orig

@st.cache
def read_data_nn():
	# Read and process data
	df_val = pd.read_csv('results/results_val.csv')
	df_orig = pd.read_csv('data/final_unscaled.csv')
	file_list = pd.read_csv('results/file_list.csv', header = None)

	file_list = set(file_list[0].str[:3].to_list())
	file_list = list(map(int, file_list))

	return df_val, file_list, df_orig


def rf_global_interpretation():
	df = pd.read_csv('data/feature_importance.csv')
	fig = px.bar(df, x="importance", y="variable", orientation='h',
             labels={'importance':'Scaled Feature Importance'}, color_discrete_sequence=["#5254a3"])
	st.plotly_chart(fig)



def rf_local_interpretation(X_train,X_test,y_train,y_test, slider_idx):
	train_features = X_train.drop(['unique_id','make','month_sold'], axis = 1)
	valid_features = X_test.drop(['unique_id','make','month_sold'], axis = 1)
	rgm = RandomForestRegressor(max_depth=5, n_estimators=50, random_state=0)
	rgm.fit(train_features, y_train)
	# rgm.score(valid_features, valid_y)

	data_for_prediction = valid_features.iloc[slider_idx,:]  # use 1 row of data here. Could use multiple rows if desired

	explainer = shap.TreeExplainer(rgm)

	# Calculate Shap values
	shap_values = explainer.shap_values(data_for_prediction)
	shap.initjs()
	shap.force_plot(np.around(explainer.expected_value, decimals = 1), np.around(shap_values, decimals=1), np.around(data_for_prediction, decimals=1), matplotlib=True, figsize=(12,3))
	st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
	plt.clf()


def rf_features_extracted(X_test, row_find, slider_idx):
	st.header("🚩 Feature Extraction")

	# 1. Colorfulness scores
	st.markdown("#### 1. Colorfulness Scores: " + str(round(row_find['score'].to_list()[0],0)))

	info_color = st.checkbox("Colorfulness scores demo")
	placeholder = st.empty()
	if info_color:
		image = Image.open('image/most_least_colorfulness.png')
		placeholder.image(image, caption='Sunrise by the mountains', use_column_width=True)
		image.close()

	# 2. Sentiments
	st.markdown("#### 2. Engine sentiments: " + str(X_test['engine_sentiment'].to_list()[slider_idx]))  ## this entire page want it to be interpretable
	st.markdown("#### 3. Bucket sentiments: " + str(X_test['bucket_sentiment'].to_list()[slider_idx]))  ## this entire page want it to be interpretable
	st.markdown("#### 4. Tires sentiments: " + str(X_test['tires_sentiment'].to_list()[slider_idx]))  ## this entire page want it to be interpretable

def rf_page(X_train,X_test,y_train,y_test, df_orig):

	############
	# side bar
	############
	st.sidebar.title("Which datapoint to explain")
	n_data = X_test.shape[0]
	slider_idx = st.sidebar.slider('Please select a datapoint', 0, n_data-1)

	row_find = df_orig[df_orig['unique_id'] == X_test['unique_id'].to_list()[slider_idx]]
	st.sidebar.markdown('**Picture**')
	st.sidebar.image('image/rf_image/'+ str(row_find['unique_id'].to_list()[0]) + '.jpg', use_column_width=True)
	st.sidebar.markdown('**Make: **' + X_test['make'].to_list()[slider_idx]) # should be a variable

	
	st.sidebar.markdown('**Age at sale: **' + str(row_find['age_at_sale'].to_list()[0]))
	st.sidebar.markdown('**Month of Sold Date: **' + row_find['month_sold'].to_list()[0])
	st.sidebar.markdown('**Engine: **' + str(row_find['engine'].to_list()[0]))
	st.sidebar.markdown('**Bucket: **' + str(row_find['bucket'].to_list()[0]))
	st.sidebar.markdown('**Tire: **' + str(row_find['tires'].to_list()[0]))

	##################
	# Features extacted
	##################
	rf_features_extracted(X_test, row_find, slider_idx)

	##################
	# Interpretation
	##################
	st.header("🚩 Global Interpretation")
	rf_global_interpretation()
	st.header("🚩 Local Interpretation")
	rf_local_interpretation(X_train,X_test,y_train,y_test, slider_idx)


def nn_page(df_results, file_list, df_orig):
	st.sidebar.title("Which datapoint to explain")
	n_data = len(file_list)
	slider_idx = st.sidebar.slider('Please select a datapoint', 0, n_data-1)
	row_find_scaled = df_results.iloc[[file_list[slider_idx]],:]
	# st.write(row_find_scaled['unique_id'].to_list()[0])
	row_find = df_orig[df_orig['unique_id'] == row_find_scaled['unique_id'].to_list()[0]]
	st.sidebar.markdown('**Make: **' + row_find['make'].to_list()[0]) # should be a variable

	
	st.sidebar.markdown('**Age at sale: **' + str(row_find['age_at_sale'].to_list()[0]))
	st.sidebar.markdown('**Month of Sold Date: **' + str(row_find['month_sold'].to_list()[0]))
	st.sidebar.markdown('**Engine: **' + str(row_find['engine'].to_list()[0]))
	st.sidebar.markdown('**Bucket: **' + str(row_find['bucket'].to_list()[0]))
	st.sidebar.markdown('**Tire: **' + str(row_find['tires'].to_list()[0]))
	return file_list[slider_idx]

def nn_interpretation(file_index):
	info_global = st.button('How it is calculated')
	if info_global:
	    st.markdown("""
	    #### How this works
	    The importance of each feature is derived from [permutation importance](https://www.kaggle.com/dansbecker/permutation-importance) -
	    by randomly shuffle a feature, how much does the model performance decrease. The ± takes into account the randomness of shuffles.
	    """)
	st.image('results/images/'+ str(file_index) + '_cam.png', use_column_width=True)
	st.image('results/images/'+ str(file_index) + '_gb.png', use_column_width=True)



def main():

	###############################################
	# Read and process tabular data (default is the first test image)
	###############################################
	model_dim = st.sidebar.selectbox('Choose a model', ('Random Forest', 'Pre-trained Neutral Net'))
	if model_dim == 'Random Forest':
		X_train, X_test, y_train, y_test, df_orig = read_split_data_rf()
		rf_page(X_train,X_test,y_train,y_test,df_orig)
	else:
		df_results, file_list, df_orig = read_data_nn()
		file_index = nn_page(df_results, file_list, df_orig)
		nn_interpretation(file_index)
		# st.write(file_index)






if __name__ == "__main__":
    main()