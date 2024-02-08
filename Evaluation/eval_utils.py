import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision.models import inception_v3
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import fid_inception_v3
import lpips
import open_clip

from compatibility_evaluator.compatibility_net import FashionEvaluator

class InceptionV3(nn.Module):
    def __init__(self, model_path, num_classes):
        super(InceptionV3, self).__init__()
        self.model = inception_v3(weights='DEFAULT')
        self.num_classes = num_classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        num_features_aux = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)

        self.model.load_state_dict(torch.load(model_path))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, x):
        x = self.model(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
    @torch.no_grad()
    def feature_extract(self, x):  # N x 3 x 299 x 299
        x = self.model._transform_input(x)
        features = self.extractor(x)  # N x 2048
        return features
    
    def extractor(self, x):
        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

class CLIPScore:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def calculate_clip_score(self, images, texts):
        img_features = self.model.encode_image(images)
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        txt_features = self.model.encode_text(texts)
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        clip_score = 100 * F.cosine_similarity(img_features, txt_features)

        del img_features
        del txt_features
        torch.cuda.empty_cache()

        return clip_score
    
    @torch.no_grad()
    def calculate_clip_img_score(self, images1, images2, similarity_func="cosine"):
        img_features1 = self.model.encode_image(images1)
        img_features1 = img_features1 / img_features1.norm(p=2, dim=-1, keepdim=True)

        img_features2 = self.model.encode_image(images2)
        img_features2 = img_features2 / img_features2.norm(p=2, dim=-1, keepdim=True)

        if similarity_func == "cosine":
            img_score = 100 * F.cosine_similarity(img_features1, img_features2)
        elif similarity_func == "euclidean":
            img_score = torch.norm(img_features1 - img_features2, p=2)  # L2-norm, Euclidean Distance
        else:
            raise ValueError(f"Unrecognized similarity function {similarity_func}.")

        del img_features1
        del img_features2
        torch.cuda.empty_cache()

        return img_score

class FIDInceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(FIDInceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        inception = fid_inception_v3()

        self.blocks = nn.ModuleList()

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def get_activations(model, data4eval, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    model.eval()

    dataloader = DataLoader(data4eval,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    
    pred_arr = np.empty((len(data4eval), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(model, data4eval, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(model, data4eval, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_given_data(gen4eval, grd4eval,
                             batch_size, dims, device, num_workers=1):
    block_idx = FIDInceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = FIDInceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(model, gen4eval, batch_size,
                                            dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(model, grd4eval, batch_size,
                                            dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    # model.to('cpu')
    del model

    return fid_value

def calculate_inception_score_given_data(data4eval, cate4eval, model_path, num_classes,
                                         batch_size, device, num_workers=1, num_splits=1,
                                         eps=1e-16, resize_input=True, normalize_input=True):
    model = InceptionV3(model_path, num_classes).to(device)

    dataloader = DataLoader(data4eval,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    
    cate_dataloader = DataLoader(cate4eval,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    
    predictions = []
    corrects = 0
    for batch, cate_batch in zip(dataloader, cate_dataloader):
        batch = batch.to(device)
        cate_batch = cate_batch.to(device)
        if resize_input:
            batch = F.interpolate(batch,
                            size=(299, 299),
                            mode='bilinear',
                            align_corners=False)
        if normalize_input:
            batch = 2 * batch - 1  # Scale from range (0, 1) to range (-1, 1)

        with torch.no_grad():
            pred = model(batch)
        pred_labels = torch.argmax(pred, dim=1)

        correct = torch.sum((pred_labels == cate_batch)).item()
        corrects += correct

        predictions.append(pred)
    
    predictions = torch.cat(predictions, dim=0)
    acc = corrects / predictions.shape[0]

    # Customized IS to evaluate DiFashion
    scores = []
    entropy = []
    uniform_prob = torch.ones(model.num_classes) / model.num_classes
    uniform_prob = uniform_prob.to(device)

    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]

        ent = - part * torch.log(part + eps)
        ent = torch.mean(torch.sum(ent, 1))
        entropy.append(ent)

        kl = part * (torch.log(part + eps) - torch.log(torch.unsqueeze(uniform_prob, 0)))
        kl = torch.mean(torch.sum(kl, 1))
        scores.append(torch.exp(kl))

    entropy = torch.stack(entropy)
    scores = torch.stack(scores)

    # model.to('cpu')
    del model

    return acc, torch.mean(entropy).item(), torch.std(entropy).item(), torch.mean(scores).item(), torch.std(scores).item()

def calculate_clip_score_given_data(img4eval, txt4eval, batch_size, device, num_workers=1):
    clip = CLIPScore(device=device)

    img_dataloader = DataLoader(img4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
        
    txt_dataloader = DataLoader(txt4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    clip_scores = []
    for img_batch, txt_batch in zip(img_dataloader, txt_dataloader):
        tokenized_txt_batch = clip.tokenizer(txt_batch).to(device)
        img_batch = img_batch.to(device)
        score = clip.calculate_clip_score(img_batch, tokenized_txt_batch)
        clip_scores.append(score)
    
    clip_scores = torch.cat(clip_scores, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return torch.mean(clip_scores).item()

def calculate_clip_img_score_given_data(gen4eval, grd4eval, batch_size, device,
                                        num_workers=1, similarity_func="cosine"):
    clip = CLIPScore(device=device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    clip_img_scores = []
    for gen_batch, grd_batch in zip(gen_dataloader, grd_dataloader):
        gen_batch = gen_batch.to(device)
        grd_batch = grd_batch.to(device)
        score = clip.calculate_clip_img_score(gen_batch, grd_batch, similarity_func=similarity_func)
        clip_img_scores.append(score)
    
    clip_img_scores = torch.cat(clip_img_scores, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return torch.mean(clip_img_scores).item()

def im2tensor_lpips(im, cent=1., factor=255./2.):
    im_np = np.array(im)
    im_tensor = torch.Tensor((im_np / factor - cent).transpose((2, 0, 1)))
    return im_tensor

@torch.no_grad()
def calculate_lpips_given_data(gen4eval, grd4eval, batch_size, device, num_workers=1, use_net="vgg"):
    model = lpips.LPIPS(net=use_net).to(device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    lpip_scores = []
    for gen_batch, grd_batch in zip(gen_dataloader, grd_dataloader):
        gen_batch = gen_batch.to(device)
        grd_batch = grd_batch.to(device)
        with torch.no_grad():
            score = model(gen_batch, grd_batch)
        lpip_scores.append(score)
    
    lpip_scores = torch.cat(lpip_scores, dim=0)

    # model.to('cpu')
    del model

    return torch.mean(lpip_scores).item()

def evaluate_personalization_given_data_sim(gen4eval, batch_size,
                                        device, num_workers=1, similarity_func="cosine"):
    # evaluate the similarity between history and generated images
    clip = CLIPScore(device=device)
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    scores = []
    with torch.no_grad():
        for gen, hist in gen_dataloader:
            gen = gen.to(device)
            hist = hist.to(device)

            gen_emb = clip.model.encode_image(gen)
            gen_emb = gen_emb / gen_emb.norm(p=2, dim=-1, keepdim=True)

            hist = hist / hist.norm(p=2, dim=-1, keepdim=True)

            if similarity_func == "cosine":
                sim_score = 100 * F.cosine_similarity(gen_emb, hist)
            elif similarity_func == "eiclidean":
                sim_score = torch.norm(gen_emb - hist, p=2)
            else:
                raise ValueError(f"Unrecognized similarity function {similarity_func}.")
            
            scores.append(sim_score)
    
    scores = torch.cat(scores, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return torch.mean(scores).item()

class CompatibilityEvaluator:
    def __init__(self, ckpt_path="", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
        self.evaluator = FashionEvaluator(cnn_feat_dim=1024)
        self.evaluator.load_state_dict(torch.load(ckpt_path))
        self.device = device

        self.clip = self.clip.to(device)
        self.evaluator = self.evaluator.to(device)
        self.clip.eval()
        self.evaluator.eval()
    
    @torch.no_grad()
    def extract_cnn_feats_gen(self, images, batch_size, num_workers):
        # extract the cnn features of generated images
        # images need to be preprocessed with open_clip trans
        img_dataloader = DataLoader(
            dataset=images,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers
        )

        cnn_feats = []
        for batch in tqdm(img_dataloader):
            batch = batch.to(self.device)
            feats = self.clip.encode_image(batch)
            cnn_feats.extend(feats.cpu().numpy())
        
        cnn_feats = np.array(cnn_feats)
        return cnn_feats
    
    @torch.no_grad()
    def evaluate_compatibility(self, outfits, cnn_feats, cnn_feats_gen):
        ofeats = []
        for olist in outfits:
            ofeat = []
            for iid in olist:
                if iid <= 0:
                    ofeat.append(cnn_feats_gen[-iid])
                else:
                    ofeat.append(cnn_feats[iid])
            ofeats.append(torch.stack(ofeat))
        ofeats = torch.stack(ofeats)
        prediction = self.evaluator(ofeats)
        scores = nn.Sigmoid()(prediction)

        return scores

def evaluate_compatibility_given_data(outfits, grd_outfits, gen4eval, cnn_feat_path, cnn_feat_gen_path, 
    ckpt_path, batch_size, device, num_workers=1):
    # outfits: iid_lists, generated_iid is negative and true_iid is positive
    # gen4eval: generated images
    model = CompatibilityEvaluator(ckpt_path, device)

    if not os.path.exists(cnn_feat_path):
        raise ValueError(f"Cnn features does not exist in {cnn_feat_path}!")
    else:
        cnn_feats = np.load(cnn_feat_path, allow_pickle=True)  # [img_num, 1024]
        cnn_feats = torch.tensor(cnn_feats).to(device)
        print(f"Successfully load cnn features of fashion images from {cnn_feat_path}")
    
    if gen4eval is None:
        cnn_feats_gen = None
    else:
        if not os.path.exists(cnn_feat_gen_path):
            print("Extract cnn features of generated images...")
            cnn_feats_gen = model.extract_cnn_feats_gen(gen4eval, batch_size, num_workers)
            # np.save(cnn_feat_gen_path, cnn_feats_gen)
            # print(f"Successfully save cnn features of generated images to {cnn_feat_gen_path}.")
        else:
            cnn_feats_gen = np.load(cnn_feat_gen_path, allow_pickle=True)
            print(f"Successfully load cnn features of generated images from {cnn_feat_gen_path}.")
        
        cnn_feats_gen = torch.tensor(cnn_feats_gen).to(device)

    outfits_loader = DataLoader(
        outfits,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )

    grd_outfits_loader = DataLoader(
        grd_outfits,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )

    scores = []
    for batch in outfits_loader:
        batch_score = model.evaluate_compatibility(batch, cnn_feats, cnn_feats_gen)
        scores.append(batch_score)
    scores = torch.cat(scores, dim=0)

    grd_scores = []
    for batch in grd_outfits_loader:
        batch_score = model.evaluate_compatibility(batch, cnn_feats, cnn_feats_gen)
        grd_scores.append(batch_score)
    grd_scores = torch.cat(grd_scores, dim=0)

    # model.clip.to('cpu')
    # model.evaluator.to('cpu')
    del model.clip
    del model.evaluator

    return torch.mean(scores).item(), torch.mean(grd_scores).item()

def calculate_clip_retrieval_acc_given_data(gen4eval, cnn_features, batch_size, device,
                                        num_workers=1, similarity_func="cosine"):
    # only return the retrieval accuracy

    clip = CLIPScore(device=device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    corrects = 0
    with torch.no_grad():
        for gen_batch, candidates in gen_dataloader:
            gen_batch = gen_batch.to(device)

            gen_batch_feats = clip.model.encode_image(gen_batch)
            gen_batch_feats = gen_batch_feats / gen_batch_feats.norm(p=2, dim=-1, keepdim=True)  # [bsz, 1024]

            candi_feats = cnn_features[candidates].to(device)
            candi_feats = candi_feats / candi_feats.norm(p=2, dim=-1, keepdim=True)  # [bsz, 5, 1024]

            sims = F.cosine_similarity(gen_batch_feats.unsqueeze(1), candi_feats, dim=-1)  # [bsz, 5]
            preds = torch.argmax(sims, dim=1)  # [bsz]

            corrects += torch.sum(preds == 0).item()
    
    acc = corrects / len(gen4eval)

    # clip.model.to('cpu')
    del clip.model

    return acc

def calculate_clip_retrieval_acc_given_data2(gen4eval, cnn_features, batch_size, device,
                                        num_workers=1, similarity_func="cosine"):
    # return the retrieval accuracy and prediction results

    clip = CLIPScore(device=device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    corrects = 0
    all_preds = []
    with torch.no_grad():
        for gen_batch, candidates in gen_dataloader:
            gen_batch = gen_batch.to(device)

            gen_batch_feats = clip.model.encode_image(gen_batch)
            gen_batch_feats = gen_batch_feats / gen_batch_feats.norm(p=2, dim=-1, keepdim=True)  # [bsz, 1024]

            candi_feats = cnn_features[candidates].to(device)
            candi_feats = candi_feats / candi_feats.norm(p=2, dim=-1, keepdim=True)  # [bsz, 5, 1024]

            sims = F.cosine_similarity(gen_batch_feats.unsqueeze(1), candi_feats, dim=-1)  # [bsz, 5]
            preds = torch.argmax(sims, dim=1)  # [bsz]

            corrects += torch.sum(preds == 0).item()
            all_preds.append(preds)
    
    acc = corrects / len(gen4eval)
    all_preds = torch.cat(all_preds, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return acc, all_preds

def clip_og_retrieval_given_data(gen4eval, grds, cnn_features, batch_size, topN, device,
                                        num_workers=1, similarity_func="cosine"):
    clip = CLIPScore(device=device)
    
    preds = []
    all_topN_iids = []
    with torch.no_grad():
        for i in range(len(gen4eval)):
            gen_img, candidates = gen4eval[i]
            gen_img = gen_img.unsqueeze(0).to(device)
            gen_feat = clip.model.encode_image(gen_img)
            gen_feat = gen_feat / gen_feat.norm(p=2, dim=-1, keepdim=True)  # [1, 1024]

            candi_feats = cnn_features[candidates].to(device)
            candi_feats = candi_feats / candi_feats.norm(p=2, dim=-1, keepdim=True)  # [cand_num, 1024]

            sims = F.cosine_similarity(gen_feat, candi_feats)  # [cand_num]
            if len(sims) > topN[-1]:
                _, pred_idxs = torch.topk(sims, topN[-1])
            else:
                _, pred_idxs = torch.topk(sims, len(sims))
                
            candidates = candidates.to(device)
            topN_iids = candidates[pred_idxs]
            all_topN_iids.append(topN_iids)
            preds.append(topN_iids[0])
    
    preds = torch.stack(preds)
    
    all_recall = []
    for i,N in enumerate(topN):
        hit = 0
        for j in range(len(grds)):
            grd = grds[j]
            if grd in all_topN_iids[j][:N]:
                hit += 1
        recall = hit / len(grds)
        all_recall.append(recall)

    # clip.model.to('cpu')
    del clip.model

    return preds, all_recall

def calculate_lpips_retrieval_acc_given_data_more(gen4eval, batch_size, device, num_workers=1, use_net="vgg"):
    model = lpips.LPIPS(net=use_net).to(device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    lpip_scores = []
    with torch.no_grad():
        for gen_batch, candidate_batch in gen_dataloader:
            gen_batch = gen_batch.to(device)
            candidate_batch = candidate_batch.to(device)
            score = model(gen_batch, candidate_batch)
            lpip_scores.append(score)
    lpip_scores = torch.cat(lpip_scores, dim=0).reshape(-1, 10, 5)

    preds = torch.argmin(lpip_scores, dim=-1)
    corrects = torch.sum(preds == 0).item()
    acc = corrects / len(lpip_scores) / 10

    # model.to('cpu')
    del model

    return acc

def calculate_lpips_retrieval_acc_given_data(gen4eval, batch_size, device, num_workers=1, use_net="vgg"):
    model = lpips.LPIPS(net=use_net).to(device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    lpip_scores = []
    with torch.no_grad():
        for gen_batch, candidate_batch in gen_dataloader:
            gen_batch = gen_batch.to(device)
            candidate_batch = candidate_batch.to(device)
            score = model(gen_batch, candidate_batch)
            lpip_scores.append(score)
    lpip_scores = torch.cat(lpip_scores, dim=0).reshape(-1, 5)

    preds = torch.argmin(lpip_scores, dim=-1)
    corrects = torch.sum(preds == 0).item()
    acc = corrects / len(lpip_scores)

    # model.to('cpu')
    del model

    return acc
