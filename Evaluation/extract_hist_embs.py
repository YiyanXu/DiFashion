import os
import random
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import open_clip
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--cnn_batch_size', type=int, default=200,
                    help='Batch size for cnn to extract features.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for compatibility net training.')
parser.add_argument('--num_workers', type=int, default=1,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--gpu', type=int, default=7,
                    help='gpu id to use')
parser.add_argument('--data_path', type=str, default='/path/to/dataset/ifashion/')
parser.add_argument('--img_folder_path', type=str, default='/path/to/imagefolder/')
parser.add_argument('--num_classes', type=int, default=50)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--log_name', type=str, default="log")


CNN_FEAT_DIM = 1024

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImagePathDataset(Dataset):
    def __init__(self, folder_path, paths, crop=None, trans=None, do_normalize=True):
        self.folder_path = folder_path
        self.paths = paths
        self.trans = trans
        self.crop = crop
        self.do_normalize = do_normalize
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')

        if self.crop is not None:
            ratio = random.random()
            if ratio < 0.5:
                img = self.crop[0](img)
            else:
                img = self.crop[1](img)

        if self.trans is not None:
            img = self.trans(img)
        if self.do_normalize:
            img = 2 * img - 1
        return img

@torch.no_grad()
def extract_cnn_features(args, img_paths, device, num_workers):
    model, _, img_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
    model = model.to(device)

    img_dataset = ImagePathDataset(args.img_folder_path, img_paths, trans=img_trans, do_normalize=False)

    img_dataloader = DataLoader(
        dataset=img_dataset,
        batch_size=args.cnn_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    cnn_feats = []
    for batch in tqdm(img_dataloader):
        batch = batch.to(device)
        feats = model.encode_image(batch)
        cnn_feats.extend(feats.cpu().numpy())
    
    cnn_feats = np.array(cnn_feats)
    return cnn_feats

def process_hist_embs(history, clip_feats):
    hist_embeds = {}
    for uid in history:
        if uid not in hist_embeds:
            hist_embeds[uid] = {}
        for cate in history[uid]:
            iids = history[uid][cate]
            hist_img_embs = clip_feats[iids]
            hist_embeds[uid][cate] = hist_img_embs.mean(dim=0)
    
    hist_embeds["null"] = clip_feats[0]

    return hist_embeds

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.gpu is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(f"cuda:{args.gpu}")
    
    print(f"Training on device {device}.")
    
    num_workers = args.num_workers

    img_paths = np.load(os.path.join(args.data_path, "all_item_image_paths.npy"), allow_pickle=True)
    max_iid = len(img_paths)

    # extract clip embeddings of all images
    cnn_feat_path = os.path.join(args.data_path, "cnn_features_clip.npy")
    if not os.path.exists(cnn_feat_path):
        print("Extract cnn features of fashion images...")
        cnn_feats = extract_cnn_features(args, img_paths, device, num_workers)
        np.save(cnn_feat_path, cnn_feats)
        print(f"Successfully save cnn features of fashion images to {cnn_feat_path}!")
    else:
        cnn_feats = np.load(cnn_feat_path, allow_pickle=True)  # [img_num, 2048]
        print(f"Successfully load cnn features of fashion images from {cnn_feat_path}")
    cnn_feats = torch.tensor(cnn_feats)
    print(f"cnn features shape: {cnn_feats.shape}")

    train_history = np.load(os.path.join(data_path, "train_history.npy"), allow_pickle=True).item()
    valid_history = np.load(os.path.join(data_path, "valid_history.npy"), allow_pickle=True).item()
    test_history = np.load(os.path.join(data_path, "test_history.npy"), allow_pickle=True).item()

    save_path = os.path.join(args.data_path, "processed")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    train_hist_embs = process_hist_embs(train_history, cnn_feats)
    np.save(os.path.join(save_path, "train_history_clipembs.npy"), np.array(train_hist_embs))
    print(f"Successfully save train_histroy_clipembs.npy to {save_path}.")

    valid_hist_embs = process_hist_embs(valid_history, cnn_feats)
    np.save(os.path.join(save_path, "valid_history_clipembs.npy"), np.array(valid_hist_embs))
    print(f"Successfully save valid_histroy_clipembs.npy to {save_path}.")

    test_hist_embs = process_hist_embs(test_history, cnn_feats)
    np.save(os.path.join(save_path, "test_history_clipembs.npy"), np.array(test_hist_embs))
    print(f"Successfully save test_histroy_clipembs.npy to {save_path}.")

if __name__ == '__main__':
    main()
