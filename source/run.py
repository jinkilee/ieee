import os
import pandas as pd
import numpy as np
from feature import make_categorical_feature_v5


if True:
	if True:
		train_transaction = pd.read_csv('/data/ieee/train_transaction.csv.zip', index_col='TransactionID', compression='zip')
		test_transaction = pd.read_csv('/data/ieee/test_transaction.csv.zip', index_col='TransactionID', compression='zip')
		train_identity = pd.read_csv('/data/ieee/train_identity.csv.zip', index_col='TransactionID', compression='zip')
		test_identity = pd.read_csv('/data/ieee/test_identity.csv.zip', index_col='TransactionID', compression='zip')
	else:
		train_transaction = pd.read_csv('/data/ieee/train_transaction_sample.csv', index_col='TransactionID')
		test_transaction = pd.read_csv('/data/ieee/test_transaction_sample.csv', index_col='TransactionID')
		train_identity = pd.read_csv('/data/ieee/train_identity_sample.csv', index_col='TransactionID')
		test_identity = pd.read_csv('/data/ieee/test_identity_sample.csv', index_col='TransactionID')
	
	train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
	test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
	del train_transaction, train_identity, test_transaction, test_identity

	### do only this one
	print('original shape: {} {}'.format(train_df.shape, test_df.shape))
	train_df, test_df = make_categorical_feature_v5(train_df, test_df)
	print('post-feature-engineering shape: {} {}'.format(train_df.shape, test_df.shape))
else:
	train_df = pd.read_csv('/data/ieee/train_preprocessed_v6.csv.zip', compression='zip')
	test_df = pd.read_csv('/data/ieee/test_preprocessed_v6.csv.zip', compression='zip')

version = make_categorical_feature_v5.__name__.split('_')[-1]
train_df.to_csv('/data/ieee/train_preprocessed_{}.csv'.format(version), index=False)
test_df.to_csv('/data/ieee/test_preprocessed_{}.csv'.format(version), index=False)
