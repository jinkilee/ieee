import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from collections import Counter

def df_size(df):
	return Counter([str(df[c].dtype) for c in df.columns])

# safe downcast
def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):
	"""
	max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
					 See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]
	avg_loss_limit - same but calculates avg throughout the series.
	na_loss_limit - not really useful.
	n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.
	"""
	is_float = str(col.dtypes)[:5] == 'float'
	na_count = col.isna().sum()
	n_uniq = col.nunique(dropna=False)
	try_types = ['float16', 'float32']

	if na_count <= na_loss_limit:
		try_types = ['int8', 'int16', 'float16', 'int32', 'float32']

	for type in try_types:
		col_tmp = col

		# float to int conversion => try to round to minimize casting error
		if is_float and (str(type)[:3] == 'int'):
			col_tmp = col_tmp.copy().fillna(fillna).round()

		col_tmp = col_tmp.astype(type)
		max_loss = (col_tmp - col).abs().max()
		avg_loss = (col_tmp - col).abs().mean()
		na_loss = np.abs(na_count - col_tmp.isna().sum())
		n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))

		if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:
			return col_tmp

	# field can't be converted
	return col


def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):
	numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
	for col in tqdm(df.columns):
		col_type = df[col].dtypes

		# collect stats
		na_count = df[col].isna().sum()
		n_uniq = df[col].nunique(dropna=False)
		
		# numerics
		if col_type in numerics:
			df[col] = sd(df[col])

		# strings
		if (col_type == 'object') and obj_to_cat:
			df[col] = df[col].astype('category')
		
		#if verbose:
		#	print('Column {}: {} -> {}, na_count={}, n_uniq={}'.format(col, col_type, df[col].dtypes, na_count, n_uniq))
		new_na_count = df[col].isna().sum()
		if (na_count != new_na_count):
			print('Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}'.format(col, col_type, df[col].dtypes, na_count, new_na_count))
		new_n_uniq = df[col].nunique(dropna=False)
		if (n_uniq != new_n_uniq):
			print('Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}'.format(col, col_type, df[col].dtypes, n_uniq, new_n_uniq))

	end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2
	percent = 100 * (start_mem - end_mem) / start_mem
	#if verbose:
	print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
	return df


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	start_mem = df.memory_usage().sum() / 1024**2	
	for col in tqdm(df.columns):
		col_type = df[col].dtypes
		if col_type in numerics:
			c_min = df[col].min()
			c_max = df[col].max()
			if str(col_type)[:3] == 'int':
				if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
					df[col] = df[col].astype(np.int8)
				elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
					df[col] = df[col].astype(np.int16)
				elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
					df[col] = df[col].astype(np.int32)
				elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
					df[col] = df[col].astype(np.int64)  
			else:
				if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
					df[col] = df[col].astype(np.float16)
				elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
					df[col] = df[col].astype(np.float32)
				else:
					df[col] = df[col].astype(np.float64)	
	end_mem = df.memory_usage().sum() / 1024**2
	if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
	return df


class NormalDistribution(object):
	def __init__(self, df, column):
		self.column = column
		self.df = df
		self.mean = self._get_mean()
		self.std = self._get_std()

	def _get_mean(self):
		if self.df[self.column].dtype == 'float16':
			self.df.loc[:, self.column] = self.df[self.column].astype(np.float32)
		return self.df[self.column].mean()

	def _get_std(self):
		if self.df[self.column].dtype == 'float16':
			self.df.loc[:, self.column] = self.df[self.column].astype(np.float32)
		return self.df[self.column].std()

	def transform(self, ser):
		return ser.apply(lambda x: (x-self.std)/self.mean)


'''
train_transaction = pd.read_csv('/data/ieee/train_transaction_sample.csv', index_col='TransactionID')
train_identity = pd.read_csv('/data/ieee/train_identity_sample.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/data/ieee/test_transaction_sample.csv', index_col='TransactionID')
test_identity = pd.read_csv('/data/ieee/test_identity_sample.csv', index_col='TransactionID')
train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


nd = NormalDistribution(train_df, 'C1')
print(train_df.C1.head(5))
train_df.loc[:, 'C1'] = nd.transform(train_df.C1)
print(train_df.C1.head(5))
'''

def feature_test(train_df, test_col, cat_cols=None):
	X = pd.DataFrame()
	y = train_df.isFraud
	
	# label encoding for categorical value
	if cat_cols:
		for c in cat_cols:
			X[c] = LabelEncoder().fit_transform(train_df[c].astype(str))
			if c in test_col:
				test_col.remove(c)
			
	# adding features
	for c in test_col:
		X[c] = train_df[c]
	
	params = {
		'objective': 'binary', 
		'boosting_type': 'gbdt', 
		'subsample': 1, 
		'bagging_seed': 11, 
		'metric': 'auc', 
		'random_state': 47
	}	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=47, stratify=y)
	clf = lgb.LGBMClassifier(**params)
	clf.fit(X_train.values, y_train)
	print('ROC AUC score with {} columns: {:.4f}'.\
		format(len(test_col), roc_auc_score(y_test, clf.predict_proba(X_test.values)[:, 1])))
	
	del X, y, X_train, X_test, y_train, y_test, clf
	gc.collect()


def minify_identity_df(df):
	df['id_12'] = df['id_12'].map({'Found':1, 'NotFound':0})
	df['id_15'] = df['id_15'].map({'New':2, 'Found':1, 'Unknown':0})
	df['id_16'] = df['id_16'].map({'Found':1, 'NotFound':0})

	df['id_23'] = df['id_23'].map({'TRANSPARENT':4, 'IP_PROXY':3, 'IP_PROXY:ANONYMOUS':2, 'IP_PROXY:HIDDEN':1})

	df['id_27'] = df['id_27'].map({'Found':1, 'NotFound':0})
	df['id_28'] = df['id_28'].map({'New':2, 'Found':1})

	df['id_29'] = df['id_29'].map({'Found':1, 'NotFound':0})

	df['id_35'] = df['id_35'].map({'T':1, 'F':0})
	df['id_36'] = df['id_36'].map({'T':1, 'F':0})
	df['id_37'] = df['id_37'].map({'T':1, 'F':0})
	df['id_38'] = df['id_38'].map({'T':1, 'F':0})

	df['id_34'] = df['id_34'].fillna(':0')
	df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)
	df['id_34'] = np.where(df['id_34']==0, np.nan, df['id_34'])
	
	df['id_33'] = df['id_33'].fillna('0x0')
	df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)
	df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)
	df['id_33'] = np.where(df['id_33']=='0x0', np.nan, df['id_33'])

	df['DeviceType'].map({'desktop':1, 'mobile':0})
	return df

