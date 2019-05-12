import argparse
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from data import Spotify_Dataset
from utils import load_model

def test_user(args):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print("using {}".format('gpu' if use_gpu else 'cpu'))
    dataset = Spotify_Dataset(args.sqlite_db, args.split, 1, data_dir=args.data_dir, 
                              use_audio_features=args.with_bias and args.model != 'mf') 
 
    model = load_model(args.model)
    model = model(dataset.n_users, dataset.n_items, args.n_features, args.n_track_features, 
        with_bias=args.with_bias).to(device) 
    model_dir = os.path.join(args.cache_dir, model.name)
    if not os.path.exists(model_dir):
        raise ValueError("{} does not exist.".format(model_dir))
    if args.test_epoch <= 0:
        raise ValueError("epoch must be > 0")
    model_path = os.path.join(model_dir,
        "epoch-{}.pkl".format(args.test_epoch))
    print("loading weights from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))

    model.eval()
    batch = dataset.get_batch_for_user(args.user_id)
    user_indices = batch['user_index'].squeeze()
    user_indices = user_indices.long().to(device)
    item_indices = batch['item_index'].squeeze()
    item_indices = item_indices.long().to(device)
    scores = batch['score'].squeeze()
    scores = scores.float().to(device)
    track_features = None
    if 'track_features' in batch:
        track_features = batch['track_features'].squeeze()
        track_features = track_features.float().to(device)
    with torch.no_grad():
        outputs = model(user_indices, item_indices, track_features=track_features).squeeze()
        probs = torch.sigmoid(outputs).data.cpu().numpy()
        preds = (probs > 0.5).astype(int)
        targets = scores.data.cpu().numpy().astype(int)
        acc = np.mean(preds == targets)
        precision = np.mean(preds[preds == 1] == targets[preds == 1])
        recall = np.mean(preds[targets == 1] == targets[targets == 1]) 
        print("user {} - precision: {:.4f} - recall: {:.4f}  acc: {:.4f}".format(
            args.user_id, precision, recall, acc))
        item_indices = item_indices.data.cpu().numpy().astype(int)
        if args.split == 'train':
            item_indices = [x.item() for x in item_indices]
        else:
            item_indices = [x.item() for x in item_indices[preds == 1]]
        item_indices = np.unique(item_indices)
        tracks = dataset.get_track_names(item_indices)
        with open("{}_{}.txt".format(args.split, args.user_id), "w") as f:
            for track in tracks:
                f.write("{} - {}\n".format(track[0],track[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Matrix Factorization Test Evaluation")
    parser.add_argument('model', help='model', choices=['mf', 'gmf', 'mlp', 'neumf', 'convncf'])
    parser.add_argument('user_id', help='user id')
    parser.add_argument('--sqlite_db', help='train data database', default='spotify.db')
    parser.add_argument('--data_dir', help='data directory', default='data/database')
    parser.add_argument('--cache_dir', help='cache directory', default='cache')
    parser.add_argument('--split', help='split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_features', help='num latent features', type=int, default=8)
    parser.add_argument('--n_track_features', help='num track features', type=int, default=15)
    parser.add_argument('--test_epoch', help='epoch to start training', type=int, default=5)
    parser.add_argument('--with_bias', help='with user/item biases', action='store_true')

    args = parser.parse_args()

    test_user(args)

