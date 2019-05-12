import argparse
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import average_precision_score
from tqdm import tqdm

from data import get_test_data
from utils import load_model

def test(args):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print("using {}".format('gpu' if use_gpu else 'cpu'))
    dataloader, dataset_size, n_users, n_items = get_test_data(
        args.sqlite_db, args.batch_size, data_dir=args.data_dir, 
        use_audio_features=args.with_bias and args.model != 'mf', num_workers=args.num_workers)

    model = load_model(args.model)
    model = model(n_users, n_items, args.n_features, args.n_track_features, 
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

    start = time.time()
    model.eval()
    num_correct = 0
    num_samples = 0
    running_loss = 0
    num_pos_pred_correct = 0
    num_pos_pred_samples = 0
    num_pos_target_correct = 0
    num_pos_target_samples = 0
    running_acc = 0
    running_precision = 0
    running_recall = 0
    progress = tqdm(dataloader, total=dataset_size)
    i = 0
    sample_probs = []
    sample_targets = []
    for batch in progress:
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
            num_correct += np.sum(preds == targets)
            num_samples += len(targets)
            running_acc = num_correct / num_samples
            num_pos_pred_correct += np.sum(preds[preds == 1] == targets[preds == 1])
            num_pos_pred_samples += len(preds[preds == 1])
            num_pos_target_correct += np.sum(preds[targets == 1] == targets[targets == 1]) 
            num_pos_target_samples += len(targets[targets == 1])
            running_precision = num_pos_pred_correct / num_pos_pred_samples if num_pos_pred_samples else 0
            running_recall = num_pos_target_correct / num_pos_target_samples
            if len(sample_targets) < 10000:
                sample_probs += list(probs)
                sample_targets += list(targets)
            
            progress.set_description(
                "test - iter {:04d} / {:04d} - precision: {:.4f} - recall: {:.4f}  acc: {:.4f}".format(
                i+1, dataset_size, running_precision, running_recall, running_acc)
            )
        i += 1

    ap = average_precision_score(sample_targets, sample_probs)
    end = time.time()
    print("AP: {:.4f}".format(ap))
    print("Finished test evaluation. Total elapsed time: {:.1f}s".format(end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Matrix Factorization Test Evaluation")
    parser.add_argument('model', help='model', choices=['mf', 'gmf', 'mlp', 'neumf', 'convncf'])
    parser.add_argument('--sqlite_db', help='train data database', default='spotify.db')
    parser.add_argument('--data_dir', help='data directory', default='data/database')
    parser.add_argument('--cache_dir', help='cache directory', default='cache')
    parser.add_argument('--batch_size', help='batch size', type=int, default=8192)
    parser.add_argument('--n_features', help='num latent features', type=int, default=8)
    parser.add_argument('--n_track_features', help='num track features', type=int, default=15)
    parser.add_argument('--test_epoch', help='epoch to start training', type=int, default=5)
    parser.add_argument('--num_workers', help='number of workers', type=int, default=4)
    parser.add_argument('--with_bias', help='with user/item biases', action='store_true')

    args = parser.parse_args()

    test(args)

