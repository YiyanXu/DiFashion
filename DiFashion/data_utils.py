import copy
import os
import random
from fileinput import filename

import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ImagePathDataset(Dataset):
    def __init__(self, folder_path, paths, trans=None, do_normalize=True):
        self.folder_path = folder_path
        self.paths = paths
        self.trans = trans
        self.do_normalize = do_normalize
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')
        if self.trans is not None:
            img = self.trans(img)
        if self.do_normalize:
            img = 2 * img - 1
        return img.to(memory_format=torch.contiguous_format).float()

class ImagePathProcess(Dataset):
    def __init__(self, folder_path, paths):
        self.folder_path = folder_path
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')
        return img

class FashionDiffusionData(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["uids"])
    
    def __getitem__(self, index):
        uids = self.data["uids"][index]
        oids = self.data["oids"][index]
        outfits = self.data["outfits"][index] 
        input_ids = self.data["input_ids"][index]
        category = self.data["category"][index]

        return {"uids": uids, "oids": oids, "outfits": outfits, 
                "input_ids": input_ids, "category": category}

class FashionFITBData(Dataset):
    def __init__(self, data, all_test_grd, fill_num=1):
        self.data = data
        self.test_grd = all_test_grd
        self.fill_num = fill_num
    
    def __len__(self):
        return len(self.data["uids"])
    
    def __getitem__(self, index):
        uids = self.data["uids"][index]
        oids = self.data["oids"][index]
        outfits = torch.tensor(self.test_grd["outfits"][index])
        for i in range(self.fill_num):
            outfits[i] = 0
        input_ids = self.data["input_ids"][index]
        category = self.data["category"][index]

        return {"uids": uids, "oids": oids, "outfits": outfits, 
                "input_ids": input_ids, "category": category}

##### Preprocessing the datasets.

def preprocess_dataset(data, data_path, id_cate_dict, history, img_dataset, tokenizer, vae, device):

    def contains_any_special_cate(category, special_cates):
        for special_cate in special_cates:
            if special_cate in category:
                return True
        return False

    # process text prompts
    def tokenize_category(data):
        data["input_ids"] = []
        for outfit_category in data["category"]:
            category_prompts = []
            for cid in outfit_category:
                category = id_cate_dict[cid]
                special_cates = ["pants", "earrings"]
                if contains_any_special_cate(category, special_cates):
                    category_prompts.append("A photo of a pair of " + category + ", on white background, high quality")
                else:
                    category_prompts.append("A photo of a " + category + ", on white background, high quality")
            inputs = tokenizer(
                category_prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            data["input_ids"].append(inputs.input_ids)
        return data

    data = tokenize_category(data)

    all_latents_path = os.path.join(data_path, "all_item_latents.npy")
    if os.path.exists(all_latents_path):
        all_latents = np.load(all_latents_path, allow_pickle=True)
        all_latents = torch.tensor(all_latents)
    else:
        vae = vae.to(device)
        batch_size = 64
        all_latents = []
        start_iid_list = list(range(0, len(img_dataset), batch_size))
        last_iid = len(img_dataset)
        with torch.no_grad():
            for start_iid in tqdm(start_iid_list):
                end_iid = start_iid + batch_size if start_iid + batch_size < last_iid else last_iid
                batch_imgs = []
                for i in range(start_iid, end_iid):
                    batch_imgs.append(img_dataset[i])
                batch_imgs = torch.stack(batch_imgs, dim=0).to(memory_format=torch.contiguous_format).float().to(device)
                batch_latents = vae.encode(batch_imgs).latent_dist.mode() * vae.config.scaling_factor
                all_latents.append(batch_latents)
            all_latents = torch.cat(all_latents, dim=0)
            all_latents = all_latents.cpu()
            np.save(all_latents_path, np.array(all_latents))

    hist_latents = {}
    for uid in history:
        if uid not in hist_latents:
            hist_latents[uid] = {}
        for cate in history[uid]:
            iids = history[uid][cate]
            hist_img_latents = all_latents[iids]
            hist_latents[uid][cate] = hist_img_latents.mean(dim=0)
    
    hist_latents["null"] = all_latents[0]
        
    outfit_category = []
    for category in data["category"]:
        category = torch.tensor(category)
        outfit_category.append(category)
    data["category"] = outfit_category

    outfits = []
    for outfit in data["outfits"]:
        outfit = torch.tensor(outfit).long()  # use as indices
        outfits.append(outfit)
    data["outfits"] = outfits
    
    return (data, hist_latents)



