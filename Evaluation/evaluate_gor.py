import os
import ast
import pathlib
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import open_clip

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

import eval_utils

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num_workers', type=int, default=1,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--gpu', type=int, default=None,
                    help='gpu id to use')
parser.add_argument('--dims4fid', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--data_path', type=str, default='/data3/xuyiy/datasets/ifashion/new_data/data_ready/')
parser.add_argument('--pretrained_evaluator_ckpt', type=str, default='./compatibility_evaluator/ifashion-ckpt/fashion_evaluator_0.0001_0.0001_test.pth')
parser.add_argument('--output_dir', type=str, default="/data3/xuyiy/fashion_output/")
parser.add_argument('--dataset', type=str, default="ifashion")
parser.add_argument('--eval_version', type=str, default="difashion-xxx")
parser.add_argument('--ckpts', type=str, default=None)
parser.add_argument('--task', type=str, default="FITB")
parser.add_argument('--num_classes', type=int, default=50)
parser.add_argument('--sim_func', type=str, default="cosine")
parser.add_argument('--lpips_net', type=str, default="vgg")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--lgor_name', type=str, default="lgor")
parser.add_argument('--mode', type=str, default="valid")
parser.add_argument('--hist_scales', type=float, default=4.0)
parser.add_argument('--mutual_scales', type=float, default=5.0)
parser.add_argument('--cate_scales', type=float, default=12.0)

SPECIAL_CATES = ["shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"]
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

class FashionEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class FashionPersonalSimDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["gen"])
    
    def __getitem__(self, index):
        gen = self.data["gen"][index]
        hist = self.data["hist"][index]

        return gen, hist

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

def cate_trans(cid, id_cate_dict):

    def contains_any_special_cate(category, special_cates):
        for special_cate in special_cates:
            if special_cate in category:
                return True
        return False
    
    category = id_cate_dict[cid]
    if contains_any_special_cate(category, SPECIAL_CATES):
        prompt = "A photo of a pair of " + category + ", on white background"
    else:
        prompt = "A photo of a " + category + ", on white background"

    return prompt

def merge_images(images):
    cols = math.ceil(math.sqrt(len(images)))
    width = images[0].width
    height = images[0].height
    total_width = width * cols
    total_height = height * cols

    merged_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    for i in range(len(images)):
        row = i // cols
        col = i % cols
        merged_image.paste(images[i], (col * width, row * height))
    
    return merged_image

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.dataset == "ifashion":
        args.data_path = '/data3/xuyiy/datasets/ifashion/new_data/data_ready/'
        args.pretrained_evaluator_ckpt = './compatibility_evaluator/ifashion-ckpt/fashion_evaluator_0.0001_0.0001_test.pth'
        args.output_dir = '/data3/xuyiy/fashion_output/'
    elif args.dataset == "polyvore":
        args.data_path = '/data3/xuyiy/datasets/polyvore/data_ready2/'
        args.pretrained_evaluator_ckpt = './compatibility_evaluator/polyvore-ckpt/fashion_evaluator_0.0001_0.0001_ratio0.1_512_finetune.pth'
        args.output_dir = '/data3/xuyiy/fashion_output_polyvore/'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}.")

    if args.mode == "valid":
        eval_path = os.path.join(args.output_dir, args.eval_version, "eval")
    else:
        eval_path = os.path.join(args.output_dir, args.eval_version, "eval-test")

    if args.ckpts != "all":
        ckpts = ast.literal_eval(args.ckpts)
    else:
        dirs = os.listdir(eval_path)
        dirs = [d.rstrip(".npy") for d in dirs if d.startswith(f"{args.task}-checkpoint") and not d.endswith("diversity.npy")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[2]))
        ckpts = [int(d.split('-')[2]) for d in dirs]
    
    # scale = f"cate{args.cate_scales}"
    scale = f"cate{args.cate_scales}-mutual{args.mutual_scales}-hist{args.hist_scales}"

    if args.gpu is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print(f"Evaluate on device {device}")

    num_workers = args.num_workers

    id_cate_dict = np.load(os.path.join(args.data_path, "new_id_cate_dict.npy"), allow_pickle=True).item()
    cid_to_label = np.load('./finetuned_inception/cid_to_label.npy', allow_pickle=True).item() # map cid to inception predicted label
    cnn_features_clip = np.load(os.path.join(args.data_path, "cnn_features_clip.npy"), allow_pickle=True)
    cnn_features_clip = torch.tensor(cnn_features_clip)

    all_image_paths = np.load(os.path.join(args.data_path, "all_item_image_paths.npy"), allow_pickle=True)
    img_folder_path = "/data3/xuyiy/datasets/ifashion/new_data/processed_images/"
    img_dataset = ImagePathProcess(img_folder_path, all_image_paths)

    if args.mode == "valid":
        history = np.load(os.path.join(args.data_path, "processed", "valid_history_clipembs.npy"), allow_pickle=True).item()
    else:
        history = np.load(os.path.join(args.data_path, "processed", "test_history_clipembs.npy"), allow_pickle=True).item()
    
    eval_save_path = os.path.join(eval_path, f"GOR_eval_results.npy")
    if not os.path.exists(eval_save_path):
        all_eval_metrics = {}
    else:
        all_eval_metrics = np.load(eval_save_path, allow_pickle=True).item()

    for ckpt in ckpts:
        gen_data = np.load(os.path.join(eval_path, f"{args.task}-checkpoint-{ckpt}-{scale}.npy"), allow_pickle=True).item()
        grd_data = np.load(os.path.join(eval_path, f"{args.task}-grd-new.npy"), allow_pickle=True).item()

        if ckpt not in all_eval_metrics:
            all_eval_metrics[ckpt] = {}
        else:
            print(f"checkpoint-{ckpt}-{scale} has already been evaluated. Skip.")
            continue

        trans = transforms.ToTensor()
        resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)
        _, _, img_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")

        gen4eval = []
        gen4clip = []
        gen_outfits_clip = []
        gen_outfits_lpips = []

        grd4eval = []
        grd4clip = []
        grd_outfits_clip = []
        grd_outfits_lpips = []

        txt4eval = []
        gen_cates = []
        for uid in gen_data:
            for oid in gen_data[uid]:
                gen_imgs = []
                for img_path in gen_data[uid][oid]["image_paths"]:
                    im = Image.open(img_path)
                    gen_imgs.append(im)

                    gen4eval.append(trans(im))
                    gen4clip.append(img_trans(im))
                
                merged_img = merge_images(gen_imgs)
                gen_outfits_clip.append(img_trans(merged_img))
                gen_outfits_lpips.append(eval_utils.im2tensor_lpips(merged_img))
                
                for cate in gen_data[uid][oid]["cates"]:
                    gen_cates.append(cid_to_label[cate.item()])
                    txt4eval.append(cate_trans(cate.item(), id_cate_dict))

                grd_imgs = []
                for img_path in grd_data[uid][oid]["image_paths"]:
                    im = Image.open(img_path)
                    if args.dataset == "polyvore":
                        im = resize(im)  # 291 --> 512
                    grd_imgs.append(im)

                    grd4eval.append(trans(im))
                    grd4clip.append(img_trans(im))
                
                merged_img = merge_images(grd_imgs)
                grd_outfits_clip.append(img_trans(merged_img))
                grd_outfits_lpips.append(eval_utils.im2tensor_lpips(merged_img))
        
        torch.cuda.empty_cache()

        # -------------------------------------------------------------- #
        #                    Calculating FID & IS                        #
        # -------------------------------------------------------------- #
        
        gen_dataset = FashionEvalDataset(gen4eval)
        grd_dataset = FashionEvalDataset(grd4eval)
        txt_dataset = FashionEvalDataset(txt4eval)
        cate_dataset = FashionEvalDataset(gen_cates)
        
        print("Calculating FID Value...")
        fid_value = eval_utils.calculate_fid_given_data(
            gen_dataset,
            grd_dataset,
            batch_size=args.batch_size,
            dims=args.dims4fid,
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        print("Calculating Inception Score...")
        is_acc, is_entropy, _, is_score, _ = eval_utils.calculate_inception_score_given_data(
            gen_dataset,
            cate_dataset,
            model_path='./finetuned_inception/Inception-finetune-epoch300',
            num_classes=args.num_classes,
            batch_size=args.batch_size, 
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["FID"] = fid_value
        all_eval_metrics[ckpt]["IS-Acc"] = is_acc
        all_eval_metrics[ckpt]["IS"] = is_score
        all_eval_metrics[ckpt]["IS-Entopy"] = is_entropy
        np.save(eval_save_path, np.array(all_eval_metrics))

        del gen4eval
        del grd4eval
        del gen_cates

        # -------------------------------------------------------------- #
        #          Calculating CLIP score and CLIP image score           #
        # -------------------------------------------------------------- #
        gen_dataset_clip = FashionEvalDataset(gen4clip)
        grd_dataset_clip = FashionEvalDataset(grd4clip)

        print("Calculating CLIP Score...")

        clip_score = eval_utils.calculate_clip_score_given_data(
            gen_dataset_clip,
            txt_dataset,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["CLIP score"] = clip_score
        np.save(eval_save_path, np.array(all_eval_metrics))

        print("Calculating Grd CLIP Score...")
        grd_clip_score = eval_utils.calculate_clip_score_given_data(
            grd_dataset_clip,
            txt_dataset,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Grd CLIP score"] = grd_clip_score
        np.save(eval_save_path, np.array(all_eval_metrics))

        del gen4clip
        del grd4clip
        del txt4eval

        gen_outfits_dataset_clip = FashionEvalDataset(gen_outfits_clip)
        grd_outfits_dataset_clip = FashionEvalDataset(grd_outfits_clip)

        print("Calculating CLIP Image Score...")
        clip_img_score = eval_utils.calculate_clip_img_score_given_data(
            gen_outfits_dataset_clip,
            grd_outfits_dataset_clip,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers,
            similarity_func=args.sim_func
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["CLIP Image score"] = clip_img_score
        np.save(eval_save_path, np.array(all_eval_metrics))
        
        del gen_outfits_clip
        del grd_outfits_clip

        # -------------------------------------------------------------- #
        #                       Calculating LPIPS                        #
        # -------------------------------------------------------------- #
        gen_outfits_dataset_lpips = FashionEvalDataset(gen_outfits_lpips)
        grd_outfits_dataset_lpips = FashionEvalDataset(grd_outfits_lpips)

        print("Calculating LPIP Score...")
        lpip_score = eval_utils.calculate_lpips_given_data(
            gen_outfits_dataset_lpips,
            grd_outfits_dataset_lpips,
            batch_size=args.batch_size // 4,
            device=device,
            num_workers=num_workers,
            use_net=args.lpips_net
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["LPIP score"] = lpip_score
        np.save(eval_save_path, np.array(all_eval_metrics))
        
        del gen_outfits_lpips
        del grd_outfits_lpips

        # -------------------------------------------------------------- #
        #                  Evaluating Personalization                    #
        # -------------------------------------------------------------- #

        print("Evaluating Personalization of similarity...")
        # The similarity between history and generated images
        gen4personal_sim = {}
        gen4personal_sim["gen"] = []
        gen4personal_sim["hist"] = []
        for uid in gen_data:
            for oid in gen_data[uid]:
                for i,img_path in enumerate(gen_data[uid][oid]["image_paths"]):
                    cate = gen_data[uid][oid]["cates"][i].item()
                    try:
                        gen4personal_sim["hist"].append(history[uid][cate])
                        im = Image.open(img_path)
                        gen4personal_sim["gen"].append(img_trans(im))
                    except:
                        continue
                        gen4personal_sim["hist"].append(history['null'])
        
        gen_dataset_personal_sim = FashionPersonalSimDataset(gen4personal_sim)
        personal_sim_score = eval_utils.evaluate_personalization_given_data_sim(
            gen4eval=gen_dataset_personal_sim,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers,
            similarity_func=args.sim_func
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Personal Sim"] = personal_sim_score
        np.save(eval_save_path, np.array(all_eval_metrics))
        
        del gen4personal_sim

        # -------------------------------------------------------------- #
        #                   Evaluating Compatibility                     #
        # -------------------------------------------------------------- #
        print("Evaluating Compatibility...")
        outfits = []
        gen_imgs = []
        gen_num = 0
        for uid in gen_data:
            for oid in gen_data[uid]:
                outfit = gen_data[uid][oid]["outfits"]  # 0 as blank to be filled
                new_outfit = []
                # real iid is positive, generated iid is negative
                for iid in outfit:
                    if iid == 0:
                        new_outfit.append(-gen_num)
                    else:
                        new_outfit.append(iid)
                gen_num += 1
                outfits.append(new_outfit)

                for img_path in gen_data[uid][oid]["image_paths"]:
                    im = Image.open(img_path)
                    gen_imgs.append(img_trans(im))

        outfits = torch.tensor(outfits)
        outfit_dataset = FashionEvalDataset(outfits)

        grd_outfits = []
        for uid in grd_data:
            for oid in grd_data[uid]:
                outfit = grd_data[uid][oid]["outfits"]
                grd_outfits.append(outfit)
        grd_outfits = torch.tensor(grd_outfits)
        grd_outfit_dataset = FashionEvalDataset(grd_outfits)

        cnn_feat_path = os.path.join(args.data_path, "cnn_features_clip.npy")
        cnn_feat_gen_path = os.path.join(eval_path, f"{args.task}-checkpoint-{ckpt}-cnnfeat.npy")
        compatibility_score, grd_compatibility_score = eval_utils.evaluate_compatibility_given_data(
            outfit_dataset,
            grd_outfit_dataset,
            gen_imgs,
            cnn_feat_path,
            cnn_feat_gen_path,
            args.pretrained_evaluator_ckpt,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers
        )
        torch.cuda.empty_cache()

        all_eval_metrics[ckpt]["Compatibility"] = compatibility_score
        all_eval_metrics[ckpt]["Grd Compatibility"] = grd_compatibility_score
        np.save(eval_save_path, np.array(all_eval_metrics))
        
        del outfits
        del grd_outfits

        print("-" * 10 + f"{args.eval_version}-checkpoint-{str(ckpt)}" + "-" * 10)
        print()
        print("## Fidelity ##")
        print(" " * 2 + f"[FID Value]: {fid_value:.2f}")
        print(" " * 2 + f"[IS Accuracy]: {is_acc:.2f}")
        print(" " * 2 + f"[Inception Score]: {is_score:.2f}")
        print(" " * 2 + f"[Inception Entropy]: {is_entropy:.2f}")
        print(" " * 2 + f"[CLIP score]: {clip_score:.2f}")
        print(" " * 2 + f"[Grd CLIP score]: {grd_clip_score:.2f}")
        print(" " * 2 + f"[CLIP image score]: {clip_img_score:.2f}")
        print(" " * 2 + f"[LPIP Score]: {lpip_score:.2f}")
        print()
        print("## Personalization ##")
        print(" " * 2 + f"[Personal Sim]: {personal_sim_score:.2f}")
        print()
        print("## Compatibility ##")
        print(" " * 2 + f"[Compatibility score]: {compatibility_score:.2f}")
        print(" " * 2 + f"[Grd Compatibility score]: {grd_compatibility_score:.2f}")
        print()
        print()

        np.save(eval_save_path, np.array(all_eval_metrics))

    print(f"All the ckpts of {args.eval_version} have been evaluated: {ckpts}")
    print(all_eval_metrics)
    print(f"Successfully saved evaluation results of {args.eval_version} checkpoint-{ckpts} to {eval_save_path}.")

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()
