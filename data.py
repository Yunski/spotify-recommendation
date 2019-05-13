import argparse
import math
import os
import time
import torch
import numpy as np
import pandas as pd

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.sql.expression import func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

class Spotify_Dataset(Dataset):
    def __init__(self, sqlite_db, split, batch_size, use_audio_features=False, data_dir=None, random_batch=False):
        if data_dir:
            sqlite_db = os.path.join(data_dir, sqlite_db)
        self.engine = create_engine('sqlite:///{}?check_same_thread=False'.format(sqlite_db))
        Base = automap_base()
        Base.prepare(self.engine, reflect=True)
        if split not in ['train', 'val', 'test']:
            raise ValueError("invalid split: must be (train/val/test)") 
        self.split = split
        self.is_train = split == 'train'
        if split == 'train':
            self.split_table_pos = Base.classes.train_pos
            self.split_table_neg = Base.classes.train_neg
        elif split == 'val':
            self.split_table = Base.classes.val
        else:
            self.split_table = Base.classes.test
        self.batch_size = batch_size
        self.track_features = Base.classes.track_features
        self.users = Base.classes.users
        self.tracks = Base.classes.tracks
        session = Session(self.engine)
        if self.is_train:
            self.pos_total = session.query(func.max(self.split_table_pos.id)).first()[0]
            self.neg_total = session.query(func.max(self.split_table_neg.id)).first()[0]
            self.total = self.pos_total + self.neg_total
            self.pos_batch_size = batch_size // 10
            self.neg_batch_size = batch_size - self.pos_batch_size
        else:
            self.total = session.query(func.max(self.split_table.id)).first()[0]
        self.n_users = session.query(func.max(self.users.id)).first()[0]
        self.n_items = session.query(func.max(self.tracks.id)).first()[0]
        self.num_batches = int(math.ceil(self.total / self.batch_size))
        self.random_batch = random_batch
        self.use_audio_features = use_audio_features

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        session = Session(self.engine) 
        rows = []
        start = idx * self.batch_size + 1
        end = min(self.total, (idx + 1) * self.batch_size)
        if self.use_audio_features:
            if self.is_train:
                rand_idx = np.random.randint(self.pos_total - self.pos_batch_size)
                pos_rows = session.query(self.split_table_pos) \
                    .options(joinedload(self.split_table_pos.track_features, innerjoin=True)) \
                    .filter(self.split_table_pos.id >= rand_idx).limit(self.pos_batch_size).all() 
                rand_idx = np.random.randint(self.neg_total - self.neg_batch_size)
                neg_rows = session.query(self.split_table_neg) \
                    .options(joinedload(self.split_table_neg.track_features, innerjoin=True)) \
                    .filter(self.split_table_neg.id >= rand_idx).limit(self.neg_batch_size).all()
                rows = pos_rows + neg_rows
            else:
                rows = session.query(self.split_table) \
                    .options(joinedload(self.split_table.track_features, innerjoin=True)) \
                    .filter(self.split_table.id >= start, self.split_table.id <= end).all() 
        else:
            if self.is_train:
                rand_idx = np.random.randint(self.pos_total - self.pos_batch_size)
                pos_rows = session.query(self.split_table_pos) \
                    .filter(self.split_table_pos.id >= rand_idx).limit(self.pos_batch_size).all() 
                rand_idx = np.random.randint(self.neg_total - self.neg_batch_size)
                neg_rows = session.query(self.split_table_neg) \
                    .filter(self.split_table_neg.id >= rand_idx).limit(self.neg_batch_size).all()
                rows = pos_rows + neg_rows
            else:
                rows = session.query(self.split_table) \
                    .filter(self.split_table.id >= start, self.split_table.id <= end).all()
        users = torch.Tensor([row.user_number for row in rows]).long()
        tracks = torch.Tensor([row.track_number for row in rows]).long()
        scores = torch.Tensor([row.score for row in rows]).long()
        batch = {
            'user_index': users,
            'item_index': tracks,
            'score': scores,
        }
        if self.use_audio_features:
            track_features = torch.FloatTensor([
                [
                    row.track_features.artist_category,
                    row.track_features.explicit,
                    row.track_features.popularity,
                    row.track_features.acousticness,
                    row.track_features.danceability,
                    row.track_features.energy,
                    row.track_features.instrumentalness,
                    row.track_features.key,
                    row.track_features.liveness,
                    row.track_features.loudness,
                    row.track_features.mode,
                    row.track_features.speechiness,
                    row.track_features.tempo,
                    row.track_features.time_signature,
                    row.track_features.valence
                ] for row in rows])
            batch['track_features'] = track_features
        return batch

    def get_batch_for_user(self, user_id):
        session = Session(self.engine) 
        rows = []
        if self.use_audio_features:
            if self.is_train:
                rows = session.query(self.split_table_pos) \
                    .options(joinedload(self.split_table_pos.track_features, innerjoin=True)) \
                    .filter(self.split_table_pos.user_id == user_id).all() 
            else:
                rows = session.query(self.split_table) \
                    .options(joinedload(self.split_table.track_features, innerjoin=True)) \
                    .filter(self.split_table.user_id == user_id).all()
        else:
            if self.is_train:
                rows = session.query(self.split_table_pos) \
                    .filter(self.split_table_pos.user_id == user_id).all()
            else:
                rows = session.query(self.split_table) \
                    .filter(self.split_table.user_id == user_id).all()
            
        users = torch.Tensor([row.user_number for row in rows]).long()
        tracks = torch.Tensor([row.track_number for row in rows]).long()
        scores = torch.Tensor([row.score for row in rows]).long()
        batch = {
            'user_index': users,
            'item_index': tracks,
            'score': scores,
        }
        if self.use_audio_features:
            track_features = torch.FloatTensor([
                [
                    row.track_features.artist_category,
                    row.track_features.explicit,
                    row.track_features.popularity,
                    row.track_features.acousticness,
                    row.track_features.danceability,
                    row.track_features.energy,
                    row.track_features.instrumentalness,
                    row.track_features.key,
                    row.track_features.liveness,
                    row.track_features.loudness,
                    row.track_features.mode,
                    row.track_features.speechiness,
                    row.track_features.tempo,
                    row.track_features.time_signature,
                    row.track_features.valence
                ] for row in rows])
            batch['track_features'] = track_features
        return batch

    def get_track_names(self, track_ids):
        session = Session(self.engine) 
        if self.is_train:
            rows = session.query(self.split_table_pos) \
                .options(joinedload(self.split_table_pos.track_features, innerjoin=True)) \
                .filter(self.split_table_pos.track_number.in_(track_ids)).all()
        else:
            rows = session.query(self.split_table) \
                .options(joinedload(self.split_table.track_features, innerjoin=True)) \
                .filter(self.split_table.track_number.in_(track_ids)).all()
        found_tracks = {
            row.track_features.track_id: (row.track_features.track_name, row.track_features.artist_name) \
            for row in rows
        }
        tracks = list(sorted([track for track in found_tracks.values()], key=lambda x: x[1])) 
        return tracks

def get_train_val_data(db_name, batch_size, use_audio_features=False, data_dir=None, num_workers=4):
    phases = ['train', 'val']
    datasets = {
        phase: Spotify_Dataset(db_name, phase, batch_size, data_dir=data_dir, 
                               use_audio_features=use_audio_features, 
                               random_batch=phase == 'train') 
        for phase in phases
    }
    samplers = {
        'train': RandomSampler,
        'val': SequentialSampler
    } 
    samplers = {
        phase: samplers[phase](datasets[phase])
        for phase in phases
    }
    dataloaders = {
        phase: DataLoader(datasets[phase], batch_size=1, 
                          sampler=samplers[phase], num_workers=num_workers)
        for phase in phases
    }
    dataset_sizes = {
        phase: len(datasets[phase])
        for phase in phases
    }
    n_users = datasets['train'].n_users
    n_items = datasets['train'].n_items
    return dataloaders, dataset_sizes, n_users, n_items

def get_test_data(db_name, batch_size, use_audio_features=False, data_dir=None, num_workers=8):
    dataset = Spotify_Dataset(db_name, 'test', batch_size, data_dir=data_dir, 
                              use_audio_features=use_audio_features) 
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=num_workers)
    return dataloader, len(dataset), dataset.n_users, dataset.n_items

def main(args):
    dataloaders, dataset_sizes, n_users, n_items = get_train_val_data(args.sqlite_db, 
        args.batch_size, use_audio_features=args.audio_features, data_dir=args.data_dir)
    for i, batch in enumerate(dataloaders[args.split]):
        for entry, v in batch.items():
            print(v[:10])
        val = input("next batch? (y/n): ")
        if val == 'y':
            continue
        else:
            break 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MSD Dataset")
    parser.add_argument('--data_dir', help='database directory', default='data/database')
    parser.add_argument('--sqlite_db', help='sqlite database name', default='spotify.db')
    parser.add_argument('--split', help='train/val/test', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--batch_size', help='batch size', type=int, default=2)
    parser.add_argument('--random_batch', help='get random batch', action='store_true')
    parser.add_argument('--audio_features', help='use audio features', action='store_true')
    args = parser.parse_args()

    main(args)

