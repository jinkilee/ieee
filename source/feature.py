import re
import pandas as pd
import numpy as np
import logging
import logging.config
import inspect
import datetime
from sklearn.preprocessing import LabelEncoder
from columns import cat_cols, match_cols, card_cols, email_cols, dev_list, info_list, count_cols, nonull_vxxx_cols, match_map
from columns import os, browser, device
from tqdm import tqdm
from utils import reduce_mem_usage_sd, reduce_mem_usage

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

