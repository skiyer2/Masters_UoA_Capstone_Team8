import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from utils.session_utils import build_sessions 

class SparseFeature():
    def __init__(self, name, vocabulary_size, embedding_dim):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        if embedding_dim == "auto":
            self.embedding_dim = 6 * int(pow(vocabulary_size, 0.25))

class DenseFeature():
    def __init__(self, name, dense_emb, embedding_dim):
        self.name = name
        if dense_emb:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 1

class SessionDataset(Dataset):
    def __init__(self, sessions, feature_names, label_name, max_session_len):
        self.sessions = sessions
        self.feature_names = feature_names
        self.label_name = label_name
        self.max_session_len = max_session_len

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session = self.sessions[idx]
        features = np.stack([row[self.feature_names].astype(float).values for row in session])
        labels = np.array([row[self.label_name] for row in session], dtype=np.float32)
        if features.shape[0] < self.max_session_len:
            pad_len = self.max_session_len - features.shape[0]
            features = np.pad(features, ((0, pad_len), (0, 0)), 'constant')
            labels = np.pad(labels, (0, pad_len), 'constant', constant_values=-1) 
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def read_data(args):
    try:
        train_data = pd.read_csv(args.data_params['data_root']+'train_.csv')
        test_data = pd.read_csv(args.data_params['data_root']+'test_.csv')
        if args.data_params['use_valid']:
            valid_data = pd.read_csv(args.data_params['data_root']+'valid_.csv')
        sparse_features = args.data_params['feature_cols']['sparse_features']['name']
        dense_features = args.data_params['feature_cols']['dense_features']['name']
    except:
        train_path = args.data_params['train_data']
        test_path = args.data_params['test_data']
        valid_path = args.data_params['valid_data']

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        if args.data_params['use_valid']:
            valid_data = pd.read_csv(valid_path)

        sparse_features = args.data_params['feature_cols']['sparse_features']['name']
        dense_features = args.data_params['feature_cols']['dense_features']['name']
        train_data[sparse_features] = train_data[sparse_features].fillna('-1', )
        train_data[dense_features] = train_data[dense_features].fillna(0, )
        test_data[sparse_features] = test_data[sparse_features].fillna('-1', )
        test_data[dense_features] = test_data[dense_features].fillna(0, )
        if args.data_params['use_valid']:
            valid_data[sparse_features] = valid_data[sparse_features].fillna('-1', )
            valid_data[dense_features] = valid_data[dense_features].fillna(0, )

        for feat in sparse_features:
            lbe = LabelEncoder()
            if args.data_params['use_valid']:
                lbe.fit(pd.concat([train_data[feat], test_data[feat], valid_data[feat]]).unique())
            else:
                lbe.fit(pd.concat([train_data[feat], test_data[feat]]).unique())
            train_data[feat] = lbe.transform(train_data[feat])
            test_data[feat] = lbe.transform(test_data[feat])
            if args.data_params['use_valid']:
                valid_data[feat] = lbe.transform(valid_data[feat])

        if args.data_params['scaler'] == 'MinMaxScaler':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()
        scaler.fit(train_data[dense_features])
        train_data[dense_features] = scaler.transform(train_data[dense_features])
        test_data[dense_features] = scaler.transform(test_data[dense_features])
        if args.data_params['use_valid']:
            valid_data[dense_features] = scaler.transform(valid_data[dense_features])

        train_data.to_csv(args.data_params['data_root']+'train_.csv', index=False)
        test_data.to_csv(args.data_params['data_root']+'test_.csv', index=False)
        if args.data_params['use_valid']:
            valid_data.to_csv(args.data_params['data_root']+'valid_.csv', index=False)

    fix_SparseFeat = [SparseFeature(feat, len(pd.concat([train_data[feat],test_data[feat],valid_data[feat]]).unique()), embedding_dim=args.model_params['embedding_dim']) for feat in sparse_features]
    fix_DenseFeat = [DenseFeature(feat, args.data_params['dense_emb'], embedding_dim=args.model_params['embedding_dim']) for feat in dense_features]
    feature_columns = fix_SparseFeat + fix_DenseFeat
    feature_names = sparse_features + dense_features
    label_col = args.data_params['label_col']['name']

    session_id_col = args.data_params.get('session_col', 'uuid') 
    time_col = args.data_params.get('time_col', 'timestamp')
    max_session_len = args.model_params.get('max_session_len', 20)

    train_sessions = build_sessions(train_data, user_col=session_id_col, time_col=time_col, max_gap=30*60*1000, max_session_len=max_session_len)
    if args.data_params['use_valid']:
        valid_sessions = build_sessions(valid_data, user_col=session_id_col, time_col=time_col, max_gap=30*60*1000, max_session_len=max_session_len)
    else:
        valid_sessions = []
    test_sessions = build_sessions(test_data, user_col=session_id_col, time_col=time_col, max_gap=30*60*1000, max_session_len=max_session_len)

    train_dataset = SessionDataset(train_sessions, feature_names, label_col, max_session_len)
    train_loader = DataLoader(train_dataset, batch_size=args.model_params['batch_size'], shuffle=True, num_workers=args.model_params['num_workers'], pin_memory=True, prefetch_factor=16)

    if args.data_params['use_valid']:
        valid_dataset = SessionDataset(valid_sessions, feature_names, label_col, max_session_len)
        valid_loader = DataLoader(valid_dataset, batch_size=args.model_params['batch_size'], shuffle=False, num_workers=args.model_params['num_workers'], pin_memory=True, prefetch_factor=16)
    else:
        valid_loader = None

    test_dataset = SessionDataset(test_sessions, feature_names, label_col, max_session_len)
    test_loader = DataLoader(test_dataset, batch_size=args.model_params['batch_size'], shuffle=False, num_workers=args.model_params['num_workers'], pin_memory=True, prefetch_factor=16)

    return train_loader, test_loader, valid_loader, fix_SparseFeat, fix_DenseFeat
