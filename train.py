import argparse
import os
import pickle
import sys
import time
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

from data import get_train_val_data
from utils import load_model

def train(args):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print("using {}".format('gpu' if use_gpu else 'cpu'))
    dataloaders, dataset_sizes, n_users, n_items = get_train_val_data(
        args.sqlite_db, args.batch_size, data_dir=args.data_dir, 
        use_audio_features=args.with_audio_feats and args.model != 'mf', num_workers=args.num_workers)
    model = load_model(args.model)
    model = model(n_users, n_items, args.n_features, args.n_track_features, 
        with_bias=args.with_bias, with_audio_feats=args.with_audio_feats).to(device) 
    model_dir = os.path.join(args.cache_dir, model.name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if args.train_epoch > 0:
        model_path = os.path.join(model_dir,
            "epoch-{}.pkl".format(args.train_epoch))
        print("loading weights from {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-3)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, 
            nesterov=True, weight_decay=1e-4)
    if args.loss == 'focal':
        criterion = FocalLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    start = time.time()
    for epoch in range(args.train_epoch, args.epochs):
        print("Epoch {}".format(epoch+1))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            num_correct = 0
            num_samples = 0
            running_loss = 0
            num_pos_pred_correct = 0
            num_pos_pred_samples = 0
            num_pos_target_correct = 0
            num_pos_target_samples = 0
            progress = tqdm(dataloaders[phase], total=dataset_sizes[phase])
            i = 0
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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(user_indices, item_indices, track_features=track_features).squeeze()
                    if phase == 'train':
                        loss = criterion(outputs, scores)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                probs = torch.sigmoid(outputs).data.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                targets = scores.data.cpu().numpy().astype(int)
                num_correct += np.sum(preds == targets)
                num_samples += len(targets)
                running_loss = (running_loss * i + loss.item()) / (i + 1)
                running_acc = num_correct / num_samples
                num_pos_pred_correct += np.sum(preds[preds == 1] == targets[preds == 1])
                num_pos_pred_samples += len(preds[preds == 1])
                num_pos_target_correct += np.sum(preds[targets == 1] == targets[targets == 1]) 
                num_pos_target_samples += len(targets[targets == 1])
                running_precision = num_pos_pred_correct / num_pos_pred_samples if num_pos_pred_samples else 0
                running_recall = num_pos_target_correct / num_pos_target_samples
                if phase == 'train':
                    progress.set_description(
                        "{} - iter {:04d} / {:04d} loss: {:.4f} - precision: {:.4f} - recall: {:.4f} - acc: {:.4f}".format(
                        phase, i+1, dataset_sizes[phase],
                        running_loss, running_precision, running_recall, running_acc)
                    )
                else:
                    progress.set_description(
                        "{} - iter {:04d} / {:04d} - precision: {:.4f} - recall: {:.4f}  acc: {:.4f}".format(
                        phase, i+1, dataset_sizes[phase], running_precision, running_recall, running_acc)
                    )
                i += 1
               
            torch.save(model.state_dict(),
            os.path.join(model_dir, "epoch-{}.pkl".format(epoch+1)))

    end = time.time()
    print("Finished training. Total elapsed time: {:.1f}s".format(end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Matrix Factorization Train")
    parser.add_argument('model', help='model', choices=['mf', 'gmf', 'mlp', 'neumf', 'convncf'])
    parser.add_argument('--sqlite_db', help='train data database', default='spotify.db')
    parser.add_argument('--data_dir', help='data directory', default='data/database')
    parser.add_argument('--cache_dir', help='cache directory', default='cache')
    parser.add_argument('--batch_size', help='batch size', type=int, default=8192)
    parser.add_argument('--epochs', help='epochs', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--n_features', help='num latent features', type=int, default=8)
    parser.add_argument('--n_track_features', help='num track features', type=int, default=15)
    parser.add_argument('--opt', help='optimizer', choices=['adam', 'rmsprop', 'sgd'], default='adam')
    parser.add_argument('--loss', help='loss', choices=['bce', 'focal'], default='bce')
    parser.add_argument('--train_epoch', help='epoch to start training', type=int, default=0)
    parser.add_argument('--num_workers', help='number of workers', type=int, default=8)
    parser.add_argument('--with_bias', help='with user/item biases', action='store_true')
    parser.add_argument('--with_audio_feats', help='with audio features', action='store_true')

    args = parser.parse_args()

    train(args)

