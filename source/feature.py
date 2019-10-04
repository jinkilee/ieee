import re
import pandas as pd
import numpy as np
import logging
import logging.config
import inspect
import datetime
from sklearn.preprocessing import LabelEncoder
from columns import cat_cols, match_cols
from tqdm import tqdm
from utils import reduce_mem_usage_sd, reduce_mem_usage

pd.set_option('max_column', 30)
logging.config.fileConfig('/anything/git/ieee/source/logging.conf')
log = logging.getLogger('ieee')
fileHandler = logging.FileHandler('/anything/git/ieee/source/log/asa_train_pretrain_model.log')
log.addHandler(fileHandler)

nearzero = 1e-4
# FIXME: import from somewhere
card_cols = ['card{}'.format(i) for i in range(1, 7)]
email_cols = ['P_emaildomain', 'R_emaildomain']
dev_list = ['chrome', 'edge', 'mobile', 'ie', 'firefox', 'safari','samsung', 'opera', 'android', 'silk', 'other', 'google', 'blu/dash', 'facebook']
info_list = ['blade', 'huawei', 'ios', 'lg', 'moto', 'rv', 'samsung', 'sm']
count_cols = ['C{}'.format(i) for i in range(1, 15)]
nonull_vxxx_cols = ['V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V297', 'V298', 'V299', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']
match_map = {
	-1: -1,
	0: 0,
	1: 1,
	2: 2,
	'T': 1,
	'F': 0,
	'M0': 0,
	'M1': 1,
	'M2': 2,
}


os = ['Windows', 'iOS', 'Android', 'Mac OS', 'Linux']
browser = ['chrome', 'mobile safari', 'ie', 'safari', 'edge', 'firefox']
device = ['Windows', 'iOS', 'MacOS', 'SM', 'SAMSUNG', 'Moto', 'LG']

def to_pattern(x: str, patterns):
	for p in patterns:
		t = re.compile('^(' + p + ').*')
		if t.match(x):
			return p
	return 'other'

def make_os_feature(df):
	return df['id_30'].map(lambda x: to_pattern(str(x), os))

def make_browser_feature(df):
	return df['id_31'].map(lambda x: to_pattern(str(x), browser))

def make_device_feature(df):
	return df['DeviceInfo'].map(lambda x: to_pattern(str(x), device))

# reference: https://www.kaggle.com/artgor/eda-and-models
def make_categorical_feature_v5(train_df, test_df):
	train_df['hour'] = train_df['TransactionDT'].map(lambda x:(x//3600)%24)
	test_df['hour'] = test_df['TransactionDT'].map(lambda x:(x//3600)%24)
	train_df['weekday'] = train_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
	test_df['weekday'] = test_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)

	test_df['os'] = make_os_feature(test_df)
	test_df['browser'] = make_browser_feature(test_df)
	test_df['device'] = make_device_feature(test_df)

	# DP1 - concatenating card1 and card2
	train_df['card1_card2'] = train_df['card1'].astype(str) + '_' + train_df['card2'].astype(str) # added at v4
	test_df['card1_card2'] = test_df['card1'].astype(str) + '_' + test_df['card2'].astype(str)	  # added at v4
	train_df['card1_productcd'] = train_df['card1'].astype(str) + '_' + train_df['ProductCD'].astype(str) # added at v4
	test_df['card1_productcd'] = test_df['card1'].astype(str) + '_' + test_df['ProductCD'].astype(str)	  # added at v4
	train_df['card1_id_31'] = train_df['card1'].astype(str) + '_' + train_df['id_31'].astype(str) # added at v4
	test_df['card1_id_31'] = test_df['card1'].astype(str) + '_' + test_df['id_31'].astype(str)	  # added at v4
	train_df['card1_id_15'] = train_df['card1'].astype(str) + '_' + train_df['id_15'].astype(str) # added at v4
	test_df['card1_id_15'] = test_df['card1'].astype(str) + '_' + test_df['id_15'].astype(str)	  # added at v4

	# DP1 - concatenating card4 and DeviceType
	train_df['card4_DeviceType'] = train_df['card4'].astype(str) + '_' + train_df['DeviceType'].astype(str) # added at v4
	test_df['card4_DeviceType'] = test_df['card4'].astype(str) + '_' + test_df['DeviceType'].astype(str)	# added at v4

	# DP1 - concatenating card4 and DeviceType
	train_df['weekday_hour'] = train_df['weekday'].astype(str) + '_' + train_df['hour'].astype(str) # added at v4
	test_df['weekday_hour'] = test_df['weekday'].astype(str) + '_' + test_df['hour'].astype(str)	# added at v4

	for c in tqdm(['card1', 'card2','ProductCD'], desc='adding count features'):
		new_col = '{}_count'.format(c)
		count_map = pd.Series(train_df[c].values.tolist() + test_df[c].values.tolist()).value_counts()
		train_df[new_col] = train_df[c].map(count_map)
		test_df[new_col] = test_df[c].map(count_map)

	for g in tqdm(['card1', 'card2', 'card3', 'card5'], desc='mean/std with cardX and TransactionAmt'):
		new_mean_col = '{}_mean_{}'.format(g, 'TransactionAmt')
		new_std_col = '{}_std_{}'.format(g, 'TransactionAmt')
		train_df[new_mean_col] = train_df['TransactionAmt']/train_df.groupby(g)['TransactionAmt'].transform('mean')
		test_df[new_mean_col] = test_df['TransactionAmt']/test_df.groupby(g)['TransactionAmt'].transform('std')
		train_df[new_std_col] = train_df['TransactionAmt']/train_df.groupby(g)['TransactionAmt'].transform('mean')
		test_df[new_std_col] = test_df['TransactionAmt']/test_df.groupby(g)['TransactionAmt'].transform('std')

	for g in tqdm(['card1', 'card2', 'card5'], desc='mean/std with cardX and id_02'):
		new_mean_col = '{}_mean_{}'.format(g, 'id_02')
		new_std_col = '{}_std_{}'.format(g, 'id_02')
		train_df[new_mean_col] = train_df['id_02']/train_df.groupby(g)['id_02'].transform('mean')
		test_df[new_mean_col] = test_df['id_02']/test_df.groupby(g)['id_02'].transform('std')
		train_df[new_std_col] = train_df['id_02']/train_df.groupby(g)['id_02'].transform('mean')
		test_df[new_std_col] = test_df['id_02']/test_df.groupby(g)['id_02'].transform('std')

	for g in ['card1', 'card2', 'card3', 'card5', 'hour', 'DeviceType']:
		for c in tqdm(['D4', 'D10', 'D15', 'id_03', 'id_04'], desc='mean/std with category features and numeric features'):
			new_mean_col = '{}_mean_{}'.format(g, c)
			new_std_col = '{}_std_{}'.format(g, c)
			train_df[new_mean_col] = train_df[c]/train_df.groupby(g)[c].transform('mean')
			test_df[new_mean_col] = test_df[c]/test_df.groupby(g)[c].transform('std')
			train_df[new_std_col] = train_df[c]/train_df.groupby(g)[c].transform('mean')
			test_df[new_std_col] = test_df[c]/test_df.groupby(g)[c].transform('std')

	for g in ['addr1', 'addr2']:
		for c in tqdm(['TransactionAmt','D4','D10','D15','id_02', 'id_03', 'id_04'], \
			desc='mean/std with addrX and numeric features'):
			if g == 'addr2' and c == 'id_02':
				continue
			new_mean_col = '{}_mean_{}'.format(g, c)
			new_std_col = '{}_std_{}'.format(g, c)
			train_df[new_mean_col] = train_df[c]/train_df.groupby(g)[c].transform('mean')
			test_df[new_mean_col] = test_df[c]/test_df.groupby(g)[c].transform('std')
			train_df[new_std_col] = train_df[c]/train_df.groupby(g)[c].transform('mean')
			test_df[new_std_col] = test_df[c]/test_df.groupby(g)[c].transform('mean')

	for g in ['browser']:
		for c in tqdm(['D4', 'id_03', 'id_04'], desc='mean/std with browser and numerical features'):
			new_mean_col = '{}_mean_{}'.format(g, c)
			new_std_col = '{}_std_{}'.format(g, c)
			train_df[new_mean_col] = train_df[c]/train_df.groupby(g)[c].transform('mean')
			test_df[new_mean_col] = test_df[c]/test_df.groupby(g)[c].transform('std')
			train_df[new_std_col] = train_df[c]/train_df.groupby(g)[c].transform('mean')
			test_df[new_std_col] = test_df[c]/test_df.groupby(g)[c].transform('std')

	# DP3 - filling Mx columns
	for c in tqdm(match_cols, desc='DP3: filling Mx columns'):
		train_df[c] = train_df[c].fillna(-1).apply(lambda x: match_map[x])
		test_df[c] = test_df[c].fillna(-1).apply(lambda x: match_map[x])

	# FE3 - weighting Mx columns
	card1_weighted_match_cols = []
	for c in tqdm(match_cols, desc='FE3: card1 weight match columns'):
		weighted_col = 'card1_weighted_{}'.format(c)
		mean_map = train_df.groupby('card1')[c].apply(lambda x: (x != -1).mean())
		train_df[weighted_col] = (train_df['card1'].map(mean_map) * train_df[c])
		test_df[weighted_col] = (test_df['card1'].fillna(-1).map(mean_map) * test_df[c])
		card1_weighted_match_cols.append(weighted_col)
	print('{} features was added'.format(len(card1_weighted_match_cols)))

	# FE4 - hour weighting Mx columns
	hour_weighted_match_cols = []
	for c in tqdm(match_cols, desc='FE4: hour weight match columns'):
		weighted_col = 'hour_weighted_{}'.format(c)
		mean_map = train_df.groupby('hour')[c].apply(lambda x: (x != -1).mean())
		train_df[weighted_col] = (train_df['hour'].map(mean_map) * train_df[c])
		test_df[weighted_col] = (test_df['hour'].fillna(-1).map(mean_map) * test_df[c])
		hour_weighted_match_cols.append(weighted_col)
	print('{} features was added'.format(len(hour_weighted_match_cols)))

	return train_df, test_df


def make_categorical_feature_v4(train_df, test_df):
	# hour,weekday were added at v4
	frequency_encoding_cols = 'ProductCD,card1,card2,card3,card5,card4,card6,addr1,hour,weekday,card4_devicetype'.split(',')
	numeric_cols = 'TransactionAmt,id_02,id_07,D4,D10,D15'.split(',')

	train_df['hour'] = train_df['TransactionDT'].map(lambda x:(x//3600)%24)
	test_df['hour'] = test_df['TransactionDT'].map(lambda x:(x//3600)%24)
	train_df['weekday'] = train_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
	test_df['weekday'] = test_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
	train_df['TransactionAmt'] = train_df['TransactionAmt'].apply(np.log1p)
	test_df['TransactionAmt'] = test_df['TransactionAmt'].apply(np.log1p)

	train_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train_df['P_emaildomain'].str.split('.', expand=True)
	train_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train_df['R_emaildomain'].str.split('.', expand=True)
	test_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test_df['P_emaildomain'].str.split('.', expand=True)
	test_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test_df['R_emaildomain'].str.split('.', expand=True)

	for c in tqdm(card_cols, desc='filling card columns'):
		try:
			train_df[c] = train_df[c].fillna(-1).astype(int)
			test_df[c] = test_df[c].fillna(-1).astype(int)
		except:
			print('exception at {}'.format(c))

	# DP1 - concatenating card1 and card2
	train_df['card1_card2'] = train_df['card1'].astype(str) + '_' + train_df['card2'].astype(str) # added at v4
	test_df['card1_card2'] = test_df['card1'].astype(str) + '_' + test_df['card2'].astype(str)	  # added at v4

	# DP1 - concatenating card4 and DeviceType
	train_df['card4_devicetype'] = \
		train_df['card4'].astype(str) + '_' + train_df['DeviceType'].astype(str) # added at v4
	test_df['card4_devicetype'] = \
		test_df['card4'].astype(str) + '_' + test_df['DeviceType'].astype(str)	 # added at v4

	# DP2 - transform TransactionAmt with np.log1p
	train_df['TransactionAmt'] = train_df['TransactionAmt'].apply(np.log1p)
	test_df['TransactionAmt'] = test_df['TransactionAmt'].apply(np.log1p)

	# FE1 - count
	frequency_encoding_features = []
	for c in tqdm(frequency_encoding_cols, desc='FE1: frequency encoding'):
		new_col = '{}_count'.format(c)
		count_map = pd.Series(train_df[c].values.tolist() + test_df[c].values.tolist()).value_counts()
		train_df.loc[:, new_col] = train_df[c].map(count_map)
		test_df.loc[:, new_col] = test_df[c].map(count_map)
		frequency_encoding_features.append(new_col)
	print('{} features was added'.format(len(frequency_encoding_features)))

	# FE2 - mean groupby columns
	frequency_transactionamt_mean_features = []
	for c1 in frequency_encoding_cols:
		for c2 in tqdm(numeric_cols, desc='FE2: mean transform for {}'.format(c1)):
			new_mean_col = '{}_mean_{}_transformed'.format(c1, c2)
			new_std_col = '{}_std_{}_transformed'.format(c1, c2)
			train_df[new_mean_col] = train_df[c2]/train_df.groupby(c1)[c2].transform('mean')
			train_df[new_std_col] = train_df[c2]/train_df.groupby(c1)[c2].transform('std')
			test_df[new_mean_col] = test_df[c2]/test_df.groupby(c1)[c2].transform('mean')
			test_df[new_std_col] = test_df[c2]/test_df.groupby(c1)[c2].transform('std')
			frequency_transactionamt_mean_features.append(new_mean_col)
			frequency_transactionamt_mean_features.append(new_std_col)
	print('{} features was added'.format(len(frequency_transactionamt_mean_features)))

	# DP3 - filling Mx columns
	for c in tqdm(match_cols, desc='DP3: filling Mx columns'):
		train_df[c] = train_df[c].fillna(-1).apply(lambda x: match_map[x])
		test_df[c] = test_df[c].fillna(-1).apply(lambda x: match_map[x])

	# FE3 - weighting Mx columns
	card1_weighted_match_cols = []
	for c in tqdm(match_cols, desc='FE3: card1 weight match columns'):
		weighted_col = 'card1_weighted_{}'.format(c)
		mean_map = train_df.groupby('card1')[c].apply(lambda x: (x != -1).mean())
		train_df[weighted_col] = (train_df['card1'].map(mean_map) * train_df[c])
		test_df[weighted_col] = (test_df['card1'].fillna(-1).map(mean_map) * test_df[c])
		card1_weighted_match_cols.append(weighted_col)
	print('{} features was added'.format(len(card1_weighted_match_cols)))

	# FE4 - hour weighting Mx columns
	hour_weighted_match_cols = []
	for c in tqdm(match_cols, desc='FE4: hour weight match columns'):
		weighted_col = 'hour_weighted_{}'.format(c)
		mean_map = train_df.groupby('hour')[c].apply(lambda x: (x != -1).mean())
		train_df[weighted_col] = (train_df['hour'].map(mean_map) * train_df[c])
		test_df[weighted_col] = (test_df['hour'].fillna(-1).map(mean_map) * test_df[c])
		hour_weighted_match_cols.append(c)
	print('{} features was added'.format(len(hour_weighted_match_cols)))

	# FE5 - week weighting Mx columns
	week_weighted_match_cols = []
	for c in tqdm(match_cols, desc='FE4: week weight match columns'):
		weighted_col = 'week_weighted_{}'.format(c)
		mean_map = train_df.groupby('weekday')[c].apply(lambda x: (x != -1).mean())
		train_df[weighted_col] = (train_df['weekday'].map(mean_map) * train_df[c])
		test_df[weighted_col] = (test_df['weekday'].fillna(-1).map(mean_map) * test_df[c])
		week_weighted_match_cols.append(c)
	print('{} features was added'.format(len(week_weighted_match_cols)))

	# DP4 - removing match columns
	train_df.drop(columns=match_cols, inplace=True)
	test_df.drop(columns=match_cols, inplace=True)
	return train_df, test_df

def make_categorical_feature_v3(train_df, test_df):
	frequency_encoding_cols = 'ProductCD,card1,card2,card3,card5,card4_card6,addr1,addr2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14'.split(',')
	train_df['hour'] = train_df['TransactionDT'].map(lambda x:(x//3600)%24)
	test_df['hour'] = test_df['TransactionDT'].map(lambda x:(x//3600)%24)
	train_df['weekday'] = train_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
	test_df['weekday'] = test_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
	train_df['TransactionAmt'] = train_df['TransactionAmt'].apply(np.log1p)
	test_df['TransactionAmt'] = test_df['TransactionAmt'].apply(np.log1p)

	train_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train_df['P_emaildomain'].str.split('.', expand=True)
	train_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train_df['R_emaildomain'].str.split('.', expand=True)
	test_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test_df['P_emaildomain'].str.split('.', expand=True)
	test_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test_df['R_emaildomain'].str.split('.', expand=True)

	for c in tqdm(card_cols, desc='filling card columns'):
		try:
			train_df[c] = train_df[c].fillna(-1).astype(int)
			test_df[c] = test_df[c].fillna(-1).astype(int)
		except:
			print('exception at {}'.format(c))

	# DP1 - concatenating card4 and card6
	train_df['card4_card6'] = train_df['card4'] + '_' + train_df['card6']
	test_df['card4_card6'] = test_df['card4'] + '_' + test_df['card6']

	# DP2 - transform TransactionAmt with np.log1p
	train_df['TransactionAmt'] = train_df['TransactionAmt'].apply(np.log1p)
	test_df['TransactionAmt'] = test_df['TransactionAmt'].apply(np.log1p)

	# FE1 - count
	frequency_encoding_features = []
	for c in tqdm(frequency_encoding_cols, desc='FE1: frequency encoding'):
		new_col = '{}_count'.format(c)
		count_map = pd.Series(train_df[c].values.tolist() + test_df[c].values.tolist()).value_counts()
		train_df.loc[:, new_col] = train_df[c].map(count_map)
		test_df.loc[:, new_col] = test_df[c].map(count_map)
		frequency_encoding_features.append(new_col)
	print('{} features was added'.format(len(frequency_encoding_features)))

	# FE2 - mean TransactionAmt
	frequency_transactionamt_mean_features = []
	for c in tqdm(frequency_encoding_cols, desc='FE2: mean transactionAmt'):
		new_col = '{}_mean_transformed'.format(c)
		sub_col = [c, 'TransactionAmt']
		mean_transformed = pd.concat([train_df[sub_col], test_df[sub_col]])\
			.groupby(c)['TransactionAmt'].transform('mean')
		train_df[new_col] = train_df['TransactionAmt'] / mean_transformed
		test_df[new_col] = test_df['TransactionAmt'] / mean_transformed
		frequency_transactionamt_mean_features.append(new_col)
	print('{} features was added'.format(len(frequency_transactionamt_mean_features)))

	# DP3 - filling Mx columns
	for c in tqdm(match_cols, desc='DP3: filling Mx columns'):
		train_df[c] = train_df[c].fillna(-1).apply(lambda x: match_map[x])
		test_df[c] = test_df[c].fillna(-1).apply(lambda x: match_map[x])

	# FE3 - weighting Mx columns
	card1_weighted_match_cols = []
	for c in tqdm(match_cols, desc='FE3: weight match columns'):
		weighted_col = 'card1_weighted_{}'.format(c)
		mean_map = train_df.groupby('card1')[c].apply(lambda x: (x != -1).mean())
		train_df[weighted_col] = (train_df['card1'].map(mean_map) * train_df[c])
		test_df[weighted_col] = (test_df['card1'].fillna(-1).map(mean_map) * test_df[c])
		card1_weighted_match_cols.append(c)
	print('{} features was added'.format(len(card1_weighted_match_cols)))


	'''
	# FE4 - fraud rate of groupby
	# for c in nonull_vxxx_cols:
	#	 a = train_df.groupby(['ProductCD', c])['isFraud'].mean()
	#	 break
	# columns whose NaN count is less than 100
	vxxx_group_count_cols = []
	vxxx_group_fraud_cols = []
	for g in ['ProductCD', 'card1']:
		for c in tqdm(nonull_vxxx_cols, desc='FE4: groupby fraud rate for {}'.format(g)):
			# fillning vxxx columns
			new_col = '{}-{}_groupby_fraudrate'.format(g, c)
			train_df[c] = train_df[c].fillna(-1)
			test_df[c] = test_df[c].fillna(-1)

			# concatenating with ProductCD
			group_col = '{}_{}'.format(g, c)
			if train_df[g].dtype == 'int64':
				train_df[group_col] = train_df[g].apply(lambda x: str(int(x))) + '_' + train_df[c].apply(lambda x: str(int(x)))
				test_df[group_col] = test_df[g].apply(lambda x: str(int(x))) + '_' + test_df[c].apply(lambda x: str(int(x)))
			else:
				train_df[group_col] = train_df[g] + '_' + train_df[c].apply(lambda x: str(int(x)))
				test_df[group_col] = test_df[g] + '_' + test_df[c].apply(lambda x: str(int(x)))

			# mapping fraud rate
			fraud_map = train_df.groupby(group_col)['isFraud'].mean()
			train_df[new_col] = train_df[group_col].map(fraud_map)
			test_df[new_col] = test_df[group_col].map(fraud_map)
			vxxx_group_fraud_cols.append(new_col)

			# mapping value count
			new_group_count_col = '{}_count'.format(group_col)
			count_map = pd.concat([train_df[group_col], test_df[group_col]]).value_counts()
			train_df[new_group_count_col] = (train_df[group_col].map(count_map)).astype(int)
			test_df[new_group_count_col] = (test_df[group_col].map(count_map)).astype(int)
			#test_df[new_group_count_col] = test_df[count_map]
			vxxx_group_count_cols.append(new_group_count_col)

			# drop temporary column
			train_df.drop(columns=group_col, inplace=True)
			test_df.drop(columns=group_col, inplace=True)
	'''

	'''
	delta_standardized_cols = []
	for c in tqdm(['D1', 'D10', 'D15'], desc='time delta standardized'):
		new_col = '{}_standardized'.format(c)
		mean = pd.concat([train_df[c], test_df[c]]).mean()
		train_df.loc[:, new_col] = train_df[c] - mean
		test_df.loc[:, new_col] = test_df[c] - mean
		delta_standardized_cols.append(new_col)
	print('{} features was added'.format(len(delta_standardized_cols)))
	'''

	'''
	#columns_a = ['TransactionAmt'] + count_cols + ['D1_standardized','D10_standardized','D15_standardized']
	columns_a = ['TransactionAmt'] + count_cols
	columns_b = ['ProductCD', 'card1', 'card4', 'card4_card6', 'addr1', 'P_emaildomain_1']
	all_df = pd.concat([train_df, test_df])
	xxx_groupby_xxx_meanstd_cols = []
	for i, col_a in enumerate(columns_a):
		for col_b in tqdm(columns_b, desc='{}/{}'.format(i+1, len(columns_a))):
			for df in [train_df, test_df]:
				df['{}_to_mean_{}'.format(col_a, col_b)] = df[col_b].map((all_df.groupby([col_b])[col_a].mean()))
				df['{}_to_std_{}'.format(col_a, col_b)] = df[col_b].map((all_df.groupby([col_b])[col_a].std() + 0.001))
				df['{}_nd_{}'.format(col_a, col_b)] = (df[col_a] - df['{}_to_mean_{}'.format(col_a, col_b)])/df['{}_to_std_{}'.format(col_a, col_b)]
				df.drop(columns=['{}_to_mean_{}'.format(col_a, col_b), '{}_to_std_{}'.format(col_a, col_b)], inplace=True)
			xxx_groupby_xxx_meanstd_cols.append('{}_nd_{}'.format(col_a, col_b))
	print('{} features was added'.format(len(xxx_groupby_xxx_meanstd_cols)))

	for c in tqdm(xxx_groupby_xxx_meanstd_cols, desc='groupby mean/std'):
		train_df[c] = train_df[c].fillna(-999).apply(lambda x: int((x*100)//1))
	test_df[c] = test_df[c].fillna(-999).apply(lambda x: int((x*100)//1))
	'''

	return train_df, test_df

def make_categorical_feature_v2(train_df, test_df, reduce_mem=True):
	fe_version = inspect.currentframe().f_code.co_name.split('_')[-1]

	train_df['hour'] = (train_df['TransactionDT'].map(lambda x:(x//3600)%24))
	test_df['hour'] = (test_df['TransactionDT'].map(lambda x:(x//3600)%24))
	train_df['weekday'] = (train_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7))
	test_df['weekday'] = (test_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7))

	# convert TransactionAmt
	train_df['TransactionAmt'] = train_df['TransactionAmt'].apply(np.log1p)

	train_df.loc[:, 'card1_card2'] = train_df['card1'].astype(str) + '_' + train_df['card2'].astype(str)
	test_df.loc[:, 'card1_card2'] = test_df['card1'].astype(str) + '_' + test_df['card2'].astype(str)

	p_email_list = ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
	r_email_list = ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
	train_df[p_email_list] = train_df['P_emaildomain'].str.split('.', expand=True)
	test_df[p_email_list] = test_df['P_emaildomain'].str.split('.', expand=True)
	train_df[r_email_list] = train_df['R_emaildomain'].str.split('.', expand=True)
	test_df[r_email_list] = test_df['R_emaildomain'].str.split('.', expand=True)
	basic_features = ['hour','weekday','card1_card2'] + p_email_list + r_email_list
	log.debug('{} features was added'.format(len(basic_features)))

	frequency_encoding_features = []
	for c in ['ProductCD', 'card1_card2', 'hour', 'P_emaildomain_1', 'R_emaildomain_2'] + card_cols:
		train_df.loc[:, c] = train_df[c].fillna(-999)
		test_df.loc[:, c] = test_df[c].fillna(-999)
		new_col = '{}_count'.format(c)
		train_df.loc[:, new_col] = (train_df[c].map(train_df[c].value_counts()))
		test_df.loc[:, new_col] = (test_df[c].map(test_df[c].value_counts()))
		frequency_encoding_features.append(c)
		frequency_encoding_features.append(new_col)
	log.debug('{} features was added'.format(len(frequency_encoding_features)))

	delta_standardized_cols = []
	for c in ['D1', 'D10', 'D15']:
		new_col = '{}_standardized'.format(c)
		mean = pd.concat([train_df[c], test_df[c]]).mean()
		train_df.loc[:, new_col] = train_df[c] - mean
		test_df.loc[:, new_col] = test_df[c] - mean
		
		delta_standardized_cols.append(new_col)
		delta_standardized_cols.append(c)
	log.debug('{} features was added'.format(len(delta_standardized_cols)))

	columns_a = ['TransactionAmt'] + count_cols + ['D1_standardized','D10_standardized','D15_standardized']
	columns_b = ['ProductCD', 'card1', 'card4', 'addr1', 'P_emaildomain_1', 'R_emaildomain_1']
	xxx_groupby_xxx_meanstd_cols = []
	for i, col_a in enumerate(columns_a):
		for col_b in tqdm(columns_b, desc='{}/{}'.format(i+1, len(columns_a))):
			for df in [train_df, test_df]:
				df['{}_to_mean_{}'.format(col_a, col_b)] = \
					(df[col_a] / (df.groupby([col_b])[col_a].transform('mean') + 0.001))
				df['{}_to_std_{}'.format(col_a, col_b)] = \
					(df[col_a] / (df.groupby([col_b])[col_a].transform('std') + 0.001))
			xxx_groupby_xxx_meanstd_cols.append('{}_to_mean_{}'.format(col_a, col_b))
			xxx_groupby_xxx_meanstd_cols.append('{}_to_std_{}'.format(col_a, col_b))
	log.debug('{} features was added'.format(len(xxx_groupby_xxx_meanstd_cols)))

	columns_a = ['addr1','addr2']
	columns_b = ['dist1','dist2']
	addr_dist_cols = []
	for i, col_a in enumerate(columns_a):
		for col_b in tqdm(columns_b, desc='{}/{}'.format(i+1, len(columns_a))):
			new_col = '{}_{}_null_rate'.format(col_a, col_b)
			for df in [train_df, test_df]:
				addr_map = pd.concat([train_df[[col_a, col_b]], test_df[[col_a,col_b]]])\
								.groupby(col_a)[col_b].apply(lambda x: x.notnull().mean())
				train_df.loc[:, new_col] = (train_df[col_a].map(addr_map))
				test_df.loc[:, new_col] = (test_df[col_a].map(addr_map))
			addr_dist_cols.append(new_col)
			addr_dist_cols.append(col_b)
			addr_dist_cols.append(col_a)
	log.debug('{} features was added'.format(len(addr_dist_cols)))

	dev_count_cols = []
	for c in ['DeviceType','DeviceInfo']:
		new_col = '{}_count_null_rate'.format(c)
		dev_count_map = pd.concat([train_df[c], test_df[c]]).value_counts()
		train_df.loc[:, new_col] = train_df[c].map(dev_count_map)
		test_df.loc[:, new_col] = test_df[c].map(dev_count_map)
		dev_count_cols.append(new_col)
		dev_count_cols.append(c)
	log.debug('{} features was added'.format(len(xxx_groupby_xxx_meanstd_cols)))

	dev_count_map = pd.concat([train_df['DeviceInfo'], test_df['DeviceInfo']]).value_counts()
	train_df.loc[:, 'DeviceInfo_count'] = train_df['DeviceInfo'].map(dev_count_map)
	test_df.loc[:, 'DeviceInfo_count'] = test_df['DeviceInfo'].map(dev_count_map)
	log.debug('{} features was added'.format(2))

	for col in match_cols:
		train_df[col] = (train_df[col].map({'T':1, 'F':0}).fillna(-999))
		test_df[col]  = (test_df[col].map({'T':1, 'F':0}).fillna(-999))

	replaced_match_cols = []
	for c in match_cols:
		new_col = '{}_replaced'.format(c)
		idx = train_df[c].isnull()
		train_df.loc[:, new_col] = train_df[c]
		train_df.loc[idx, new_col] = train_df[idx]['ProductCD']
		idx = test_df[c].isnull()
		test_df.loc[:, new_col] = test_df[c]
		test_df.loc[idx, new_col] = test_df[idx]['ProductCD']
		replaced_match_cols.append(new_col)
	log.debug('{} features was added'.format(len(replaced_match_cols)))

	fe_features = list(
		set(basic_features) | \
		set(frequency_encoding_features) | \
		set(replaced_match_cols) | \
		set(dev_count_cols) | \
		set(addr_dist_cols) | \
		set(xxx_groupby_xxx_meanstd_cols) | \
		set(delta_standardized_cols) | \
		set(frequency_encoding_features)
	)

	# do label-encoding
	for c in tqdm(fe_features, desc='label-encoding'):
		if c == 'isFraud':
			continue
		if train_df[c].dtype=='object' or test_df[c].dtype=='object':
			train_df[c] = train_df[c].fillna('NA')
			test_df[c] = test_df[c].fillna('NA')
			le = LabelEncoder()
			le.fit(list(train_df[c].values) + list(test_df[c].values))
			train_df[c] = le.transform(list(train_df[c].values))
			test_df[c] = le.transform(list(test_df[c].values))

	'''
	# safe memory reduction
	if reduce_mem:
		train_df = reduce_mem_usage_sd(train_df, verbose=True)
		test_df  = reduce_mem_usage_sd(test_df, verbose=True)
	'''

	return train_df, test_df, fe_version

def make_categorical_feature_v1(train_df, test_df):
	fe_version = inspect.currentframe().f_code.co_name.split('_')[-1]
	log.info('original train/test shape: {} {}'.format(train_df.shape, test_df.shape))

	# prepare entire dataset
	df = pd.concat([train_df, test_df])

	# convert TransactionAmt
	train_df['TransactionAmt'] = train_df['TransactionAmt'].apply(np.log1p)

	# convert TransactionDT
	train_df['hour'] = train_df['TransactionDT'].map(lambda x:(x//3600)%24)
	test_df['hour'] = test_df['TransactionDT'].map(lambda x:(x//3600)%24)
	train_df['weekday'] = train_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)
	test_df['weekday'] = test_df['TransactionDT'].map(lambda x:(x//(3600 * 24))%7)

	# split email_cols
	#train_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train_df['P_emaildomain'].str.split('.', expand=True)
	#train_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train_df['R_emaildomain'].str.split('.', expand=True)
	#test_df[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test_df['P_emaildomain'].str.split('.', expand=True)
	#test_df[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test_df['R_emaildomain'].str.split('.', expand=True)
	#train_df.drop(columns=['P_emaildomain', 'R_emaildomain'], inplace=True)
	#test_df.drop(columns=['P_emaildomain', 'R_emaildomain'], inplace=True)

	# groupby `TransactionDT` for count
	cols_to_count = 'ProductCD,card1,card2,card3,card4,card5,card6,addr1,addr2,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14'
	for col in tqdm(cols_to_count.split(','), desc='groupby TransactionDT for count'):
		col_count = df.groupby(col)['TransactionDT'].count()
		train_df[col+'_count'] = train_df[col].map(col_count)
		test_df[col+'_count'] = test_df[col].map(col_count)

	# groupby `TransactionAmt` for mean
	cols_to_mean = 'card1,card2,card5,addr1,addr2'
	for c in tqdm(cols_to_mean.split(','), desc='groupby TransactionAmt for mean'):
		col_mean = df.groupby(c)['TransactionAmt'].mean()
		train_df[c+'_amtmean'] = train_df[c].map(col_mean)
		test_df[c+'_amtmean'] = test_df[c].map(col_mean)
		col_count1 = df[df['C5'] == 0].groupby(c)['C5'].count()
		col_count2 = df[df['C5'] != 0].groupby(c)['C5'].count()
		train_df[c+'_C5count'] = train_df[c].map(col_count2) / (train_df[c].map(col_count1) + 0.01)
		test_df[c+'_C5count'] = test_df[c].map(col_count2) / (test_df[c].map(col_count1) + 0.01)

	# do groupby-normalizing
	for gc in tqdm(['ProductCD'] + card_cols, desc='groupby-normalizing'):
		for c in ['TransactionAmt','C1','C2','C6','C11','C13','C14']:	# hand-picked features
			train_df, test_df = groupby_normalize(
				data_df=train_df, 
				groupby_columns=gc,
				target=c, 
				train_df=train_df, 
				test_df=test_df)
	log.info('after groupby_normalize(), train/test shape: {} {}'.format(train_df.shape, test_df.shape))

	'''
	#columns_a = ['TransactionAmt'] + count_cols
	#columns_b = ['ProductCD', 'card1', 'card4', 'addr1', 'P_emaildomain_1', 'R_emaildomain_1']
	columns_a = ['TransactionAmt']
	columns_b = ['ProductCD']
	xxx_groupby_xxx_meanstd_cols = []
	for i, col_a in enumerate(columns_a):
		for col_b in tqdm(columns_b, desc='{}/{}'.format(i+1, len(columns_a))):
			for df in [train_df, test_df]:
				df['{}_to_mean_{}'.format(col_a, col_b)] = \
					(df[col_a] / (df.groupby([col_b])[col_a].transform('mean') + 0.001))
				df['{}_to_std_{}'.format(col_a, col_b)] = \
					(df[col_a] / (df.groupby([col_b])[col_a].transform('std') + 0.001))
			xxx_groupby_xxx_meanstd_cols.append('{}_to_mean_{}'.format(col_a, col_b))
			xxx_groupby_xxx_meanstd_cols.append('{}_to_std_{}'.format(col_a, col_b))
	log.debug('{} features was added'.format(len(xxx_groupby_xxx_meanstd_cols)))
	'''

	return train_df, test_df, fe_version

def get_mean_std_groupby(df, groupby_column, target):
	new_col = '_{}'.format(target)
	df.loc[:, new_col] = df[target].apply(np.log1p)
	#mean = df.groupby(groupby_column)[new_col].transform('mean')
	#std = df.groupby(groupby_column)[new_col].transform('std')
	mean = df.groupby(groupby_column)[new_col].mean()
	std = df.groupby(groupby_column)[new_col].std()
	df.drop(columns=[new_col], inplace=True)
	return mean, std

def normalize_dataframe(df, target, mean, std):
	return (df[target].apply(np.log1p) - df[mean])/df[std]
	
def groupby_normalize(data_df, groupby_columns, target, train_df=None, test_df=None):
	# split by isFraud
	fraud_df = data_df[data_df.isFraud == 1]
	normal_df = data_df[data_df.isFraud == 0]

	# define new column name
	new_fraud_std_col = 'std_{}_fraud'.format(target)
	new_normal_std_col = 'std_{}_normal'.format(target)
	new_fraud_mean_col = 'mean_{}_fraud'.format(target)
	new_normal_mean_col = 'mean_{}_normal'.format(target)
	new_fraud_col = 'nd_{}_fraud'.format(target)
	new_normal_col = 'nd_{}_normal'.format(target)

	# calculate groupby mean and std for fraud_df/normal_df
	fraud_mean, fraud_std = get_mean_std_groupby(fraud_df, groupby_columns, target)
	normal_mean, normal_std = get_mean_std_groupby(normal_df, groupby_columns, target)

	if isinstance(train_df, pd.DataFrame):
		# prepare mean and std for train_df and test_df
		train_df.loc[:, new_fraud_mean_col] = train_df[groupby_columns].map(fraud_mean)
		train_df.loc[:, new_fraud_std_col] = train_df[groupby_columns].map(fraud_std)
		train_df.loc[:, new_normal_mean_col] = train_df[groupby_columns].map(normal_mean)
		train_df.loc[:, new_normal_std_col] = train_df[groupby_columns].map(normal_std)
	
		# normalize train_df and test_df
		train_df.loc[:, new_fraud_col] = normalize_dataframe(train_df, target, new_fraud_mean_col, new_fraud_std_col)
		train_df.loc[:, new_normal_col] = normalize_dataframe(train_df, target, new_normal_mean_col, new_normal_std_col)
		log.debug('{} has been added on train_df'.format(new_fraud_col))
		log.debug('{} has been added on train_df'.format(new_normal_col))

	if isinstance(test_df, pd.DataFrame):
		# prepare mean and std for train_df and test_df
		test_df.loc[:, new_fraud_mean_col] = test_df[groupby_columns].map(fraud_mean)
		test_df.loc[:, new_fraud_std_col] = test_df[groupby_columns].map(fraud_std)
		test_df.loc[:, new_normal_mean_col] = test_df[groupby_columns].map(normal_mean)
		test_df.loc[:, new_normal_std_col] = test_df[groupby_columns].map(normal_std)

		# normalize train_df and test_df
		test_df.loc[:, new_fraud_col] = normalize_dataframe(test_df, target, new_fraud_mean_col, new_fraud_std_col)
		test_df.loc[:, new_normal_col] = normalize_dataframe(test_df, target, new_normal_mean_col, new_normal_std_col)
		log.debug('{} has been added on test_df'.format(new_fraud_col))
		log.debug('{} has been added on test_df'.format(new_normal_col))
	return train_df, test_df

