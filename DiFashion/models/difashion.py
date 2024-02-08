import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
import tqdm
from typing import Callable, List, Optional, Union
import PIL
from diffusers import (AutoencoderKL, UNet2DConditionModel, PNDMScheduler)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils.import_utils import is_xformers_available, is_accelerate_available, is_accelerate_version
from diffusers.utils.torch_utils import randn_tensor
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class MutualEncoder(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, cate_num, cate_emb_size, latent_channels, latent_size, hid_dim):
        super().__init__()
        self.category_embedding = nn.Embedding(cate_num, cate_emb_size)  # useless embedding
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels * latent_size * latent_size, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_dim, latent_channels * latent_size * latent_size),
            nn.Tanh()  # restrict the output in [-1., 1.]
        )
    
    def forward(self, mutual_emb):
        bsz = mutual_emb.shape[0]
        mutual_emb = mutual_emb.view(bsz, -1)
        mutual_guidance = self.mlp(mutual_emb)
        mutual_guidance = mutual_guidance.view(bsz, self.latent_channels,
                                               self.latent_size, self.latent_size)

        return mutual_guidance

class DiFashion(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True 

    @register_to_config
    def __init__(
        self,
        args,
        logger,
        cate_num,
        device
    ):
        super(DiFashion, self).__init__()
        self.args = args
        self.logger = logger

        logger.info("load PDNMScheduler...")
        self.noise_scheduler = PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("load CLIPTokenizer...")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        logger.info("load CLIPTextModel...")
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        logger.info("load VAE...")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        logger.info("load UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
        )

        self.logger.info("Initializing the DiFashion UNet from the pretrained UNet.")
        # Extend the first convolutional layer of the pretrained UNet for history condition.
        in_channels = 8  # [latents, history_latents]
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride, self.unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in
        
        self.fashion_encoder = MutualEncoder(
            cate_num=cate_num, 
            cate_emb_size=args.category_emb_size,
            latent_channels=self.vae.config.latent_channels,  # 4
            latent_size=self.unet.config.sample_size,  # 64
            hid_dim=args.hid_dim
        )
        self.fashion_encoder.apply(xavier_normal_initialization)

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    def forward(self, batch, img_dataset, history, null_img, mask_ratio, coupling_mask_ratio, cate_mask_ratio, weight_dtype, generator):
        uids = batch["uids"]
        outfits = batch["outfits"]
        category = batch["category"]  ### outfit_category: [cate_1, cate_2, ..., cate_n]
        input_ids = batch["input_ids"]  ### prompts: [prompt_1, prompt_2, ..., prompt_n]
        ### user_history: [{"cate_1":hist_latent, "cate_2":hist_latent, ...}, {...}, ...]
        null_img = null_img.unsqueeze(0)
        null_latent = self.vae.encode(null_img.to(weight_dtype)).latent_dist.mode()[0]
        null_latent = null_latent * self.vae.config.scaling_factor

        outfit_images = []
        bsz = len(uids)
        olen = len(outfits[0])
        if olen < 0:
            print(outfits)
            print(olen)
            raise ValueError
        for i in range(len(uids)):
            for iid in outfits[i]:
                outfit_images.append(img_dataset[iid])
        outfit_images = torch.stack(outfit_images).to(self.device)  # [bsz * 4, 3, 512, 512]

        latents = self.vae.encode(outfit_images.to(weight_dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # [bsz * 4, 3, 64, 64]

        noise = torch.randn_like(latents)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.repeat_interleave(olen)  # outfit_length=4
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        if self.args.use_mutual_guidance:
            mutual_cond = []
            for i,idx in enumerate(range(0, bsz * olen, olen)):
                weights = torch.ones(olen, olen).masked_fill((torch.eye(olen) > 0), 0.).to(self.device)
                weights = weights / torch.sum(weights, dim=1)

                mutual_latents = noisy_latents[idx:idx+olen]
                for weight in weights:
                    weighted_mutual_latent = sum([w * emb for w,emb in zip(weight, mutual_latents)])
                    mutual_cond.append(weighted_mutual_latent)
            mutual_cond = torch.stack(mutual_cond).to(self.device, dtype=weight_dtype)
            mutual_cond = self.fashion_encoder(mutual_cond)
        else:
            mutual_cond = torch.stack([null_latent] * (bsz * olen))

        assert mutual_cond.shape == noisy_latents.shape

        hist_latents = []
        for i in range(bsz):
            for cate in category[i]:
                if self.args.use_history and cate in history[uids[i].item()]:
                    hist_latents.append(history[uids[i].item()][cate])
                else:
                    hist_latents.append(null_latent)
        hist_latents = torch.stack(hist_latents).to(self.device)

        masked_mutual_cond = mutual_cond.clone()
        if mask_ratio is not None:
            random_p = torch.rand(bsz * olen, device=self.device, generator=generator)
            if self.args.use_history and self.args.use_mutual_guidance:
                image_mask = (
                    random_p < mask_ratio + coupling_mask_ratio
                )
                if image_mask.sum() > 0:
                    hist_latents[image_mask] = torch.stack([null_latent] * image_mask.sum())

                mutual_mask = (
                    (random_p >= mask_ratio)
                    & (random_p < 2 * mask_ratio + coupling_mask_ratio)
                )
                if mutual_mask.sum() > 0:
                    masked_mutual_cond[mutual_mask] = torch.stack([null_latent] * mutual_mask.sum())
            elif self.args.use_history:
                image_mask = (
                    random_p < mask_ratio
                )
                if image_mask.sum() > 0:
                    hist_latents[image_mask] = torch.stack([null_latent] * image_mask.sum())
            elif self.args.use_mutual_guidance:
                mutual_mask = (
                    random_p < mask_ratio
                )
                if mutual_mask.sum() > 0:
                    masked_mutual_cond[mutual_mask] = torch.stack([null_latent] * mutual_mask.sum())
        
        added_noisy_latents = (1 - self.args.eta) * noisy_latents + self.args.eta * masked_mutual_cond
        added_noisy_latents = torch.cat([added_noisy_latents, hist_latents], dim=1)

        all_input_ids = []
        for i in range(bsz):
            for input_id in input_ids[i]:
                all_input_ids.append(input_id)
        all_input_ids = torch.stack(all_input_ids).to(self.device)

        encoder_hidden_states = self.text_encoder(all_input_ids)[0]

        null_input_ids = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        null_input_ids = null_input_ids.to(self.device)
        null_prompt = self.text_encoder(null_input_ids)[0]
        if cate_mask_ratio is not None:
            random_p = torch.rand(bsz * olen, device=self.device, generator=generator)
            cate_mask = (random_p < cate_mask_ratio)
            if cate_mask.sum() > 0:
                encoder_hidden_states[cate_mask] = torch.cat([null_prompt] * cate_mask.sum())

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            ### Page5 https://arxiv.org/pdf/2202.00512.pdf
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet(
            added_noisy_latents,
            timesteps, 
            encoder_hidden_states
        ).sample
        
        if self.args.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = self.compute_snr(timesteps)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

            mse_loss_weights = (
                torch.stack([snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        
        return loss

    def pred_ori_sample_given_epsilon(self, timestep, noisy_latent, epsilon):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_prod_t = alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_ori_sample = (noisy_latent - beta_prod_t ** (0.5) * epsilon) / alpha_prod_t ** (0.5)

        return pred_ori_sample.clamp(-1., 1.)
    
    @torch.no_grad()
    def fashion_generation(
        self,
        uids: torch.Tensor = None,  # [bsz,]
        oids: torch.Tensor = None,  # [bsz,]
        input_ids: Optional[torch.FloatTensor] = None,  # [bsz, 4, 77]
        olists: torch.Tensor = None,  # [bsz, 4]
        outfit_images: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,  # [bsz, 4, 3, 512, 512]
        category: List[int] = None,  # [bsz, 4]
        history: dict = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        category_guidance_scale: float = 7.5,
        hist_guidance_scale: float = 7.5,
        mutual_guidance_scale: float = 7.5,
        null_img: torch.FloatTensor = None,
        eta: float = 0.0,
        init_latents: torch.Tensor = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        if self.args.use_history and hist_guidance_scale > 1.0:
            do_classifier_free_guidance_hist = True
        else:
            do_classifier_free_guidance_hist = False
        if self.args.use_mutual_guidance and mutual_guidance_scale > 1.0:
            do_classifier_free_guidance_mutual = True
        else:
            do_classifier_free_guidance_mutual = False
        if category_guidance_scale > 1.0:
            do_classifier_free_guidance_category = True
        else:
            do_classifier_free_guidance_category = False

        if do_classifier_free_guidance_hist and do_classifier_free_guidance_mutual and do_classifier_free_guidance_category:
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance = False

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        bsz = olists.shape[0]
        olen = olists.shape[1]
        fill_idx = torch.nonzero(olists == 0)
        fill_num = fill_idx.shape[0]
        fill_cate = category[fill_idx[:, 0], fill_idx[:, 1]]
        fill_uids = uids[fill_idx[:, 0]]
        fill_oids = oids[fill_idx[:, 0]]
        full_cate = category[fill_idx[:, 0]]

        fill_input_ids = input_ids[fill_idx[:, 0], fill_idx[:, 1]]
        category_prompts = self.text_encoder(
            fill_input_ids.to(self.device),
        )[0].to(dtype=self.text_encoder.dtype, device=self.device)

        null_input_ids = self.tokenizer(
            [""],
            padding="max_length",
            max_length=category_prompts.shape[1],
            truncation=True,
            return_tensors="pt",
        ).input_ids
        null_input_ids = null_input_ids.to(self.device)
        null_prompt = self.text_encoder(null_input_ids)[0]
        null_prompts = torch.cat([null_prompt] * category_prompts.shape[0], dim=0)

        # Set timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        if init_latents is None:
            latents = self.prepare_latents(
                fill_num,
                num_channels_latents,
                height,
                width,
                category_prompts.dtype,
                self.device,
                generator
            )
            init_latents = latents.clone()
        else:
            latents = init_latents.clone()  # designated initial latents

        null_img = null_img.unsqueeze(0)
        null_latent = self.vae.encode(null_img).latent_dist.mode()[0] * self.vae.config.scaling_factor

        # Prepare history latents
        hist_latents = []
        for i,cate in enumerate(fill_cate):
            uid = uids[fill_idx[i][0]].item()
            if self.args.use_history and cate in history[uid]:
                hist_latents.append(history[uid][cate])
            else:
                hist_latents.append(null_latent)
        hist_latents = torch.stack(hist_latents).to(self.device)
        
        if do_classifier_free_guidance:
            null_hist_latents = torch.stack([null_latent] * hist_latents.shape[0])
            hist_latents = torch.cat([hist_latents, null_hist_latents, null_hist_latents, null_hist_latents], dim=0)
        elif do_classifier_free_guidance_category:
            if do_classifier_free_guidance_hist:
                null_hist_latents = torch.stack([null_latent] * hist_latents.shape[0])
                hist_latents = torch.cat([hist_latents, null_hist_latents, null_hist_latents], dim=0)
            elif do_classifier_free_guidance_mutual:
                hist_latents = torch.cat([hist_latents] * 3, dim=0)
            else:
                hist_latents = torch.cat([hist_latents] * 2, dim=0)
        else:
            if do_classifier_free_guidance_hist:
                null_hist_latents = torch.stack([null_latent] * hist_latents.shape[0])
                hist_latents = torch.cat([hist_latents, null_hist_latents], dim=0)
            elif do_classifier_free_guidance_mutual:
                hist_latents = torch.cat([hist_latents] * 2, dim=0)
            else:
                pass
        
        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [category_prompts, category_prompts, category_prompts, null_prompts], dim=0
            ).to(dtype=self.text_encoder.dtype, device=self.device)
        elif do_classifier_free_guidance_category:
            if do_classifier_free_guidance_hist or do_classifier_free_guidance_mutual:
                encoder_hidden_states = torch.cat(
                    [category_prompts, category_prompts, null_prompts], dim=0
                ).to(dtype=self.text_encoder.dtype, device=self.device)
            else:
                encoder_hidden_states = torch.cat(
                    [category_prompts, null_prompts], dim=0
                ).to(dtype=self.text_encoder.dtype, device=self.device)
        else:
            if do_classifier_free_guidance_hist or do_classifier_free_guidance_mutual:
                encoder_hidden_states = torch.cat(
                    [category_prompts] * 2, dim=0
                ).to(dtype=self.text_encoder.dtype, device=self.device)
            else:
                encoder_hidden_states = category_prompts
        
        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.noise_scheduler.order

        all_latents = self.vae.encode(
            outfit_images
        ).latent_dist.mode() * self.vae.config.scaling_factor

        gen_masks = (olists == 0)
        mutual_indicies = []
        all_num = 0
        for i,_ in enumerate(olists):
            gen_mask = gen_masks[i]
            gen_num = sum(gen_mask)
            indicies = torch.arange(olen) + i * olen
            indicies[gen_mask] = -torch.arange(all_num, all_num + gen_num) - 1
            
            mutual_indicies.append(indicies)
            all_num += gen_num
        mutual_indicies = torch.stack(mutual_indicies).to(self.device)
        assert all_num == fill_idx.shape[0]
        
        prev_latents = latents.clone().to(dtype=null_latent.dtype)

        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance.
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 4)
            elif do_classifier_free_guidance_category:
                if do_classifier_free_guidance_mutual or do_classifier_free_guidance_hist:
                    latent_model_input = torch.cat([latents] * 3)
                else:
                    latent_model_input = torch.cat([latents] * 2)
            else:
                if do_classifier_free_guidance_mutual or do_classifier_free_guidance_hist:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

            # concat latents, hist_latents
            scaled_latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            # Prepare mutual guidance
            if self.args.use_mutual_guidance:
                mutual_cond = []
                for j,(o_idx, i_idx) in enumerate(fill_idx):
                    weights = torch.ones(olen).to(self.device)
                    weights[i_idx] = 0.

                    mutual_latents = torch.zeros_like(null_latent).unsqueeze(0).repeat(olen, 1, 1, 1).to(self.device)
                    gen_mask = gen_masks[o_idx]
                    mutual_latents[~gen_mask] = all_latents[mutual_indicies[o_idx][~gen_mask]]
                    mutual_latents[gen_mask] = prev_latents[-mutual_indicies[o_idx][gen_mask]-1]
                    
                    weighted_latents = sum([weight * emb for weight,emb in zip(weights, mutual_latents)])
                    mutual_cond.append(weighted_latents)
                
                mutual_cond = torch.stack(mutual_cond).to(self.device)
                mutual_cond = self.fashion_encoder(mutual_cond)
            else:
                mutual_cond = torch.stack([null_latent] * fill_num).to(self.device)

            if do_classifier_free_guidance:
                null_mutual_cond = torch.stack([null_latent] * mutual_cond.shape[0])
                mutual_cond = torch.cat([mutual_cond, mutual_cond, null_mutual_cond, null_mutual_cond], dim=0)
            elif do_classifier_free_guidance_category:
                if do_classifier_free_guidance_mutual:
                    null_mutual_cond = torch.stack([null_latent] * mutual_cond.shape[0])
                    mutual_cond = torch.cat([mutual_cond, null_mutual_cond, null_mutual_cond], dim=0)
                elif do_classifier_free_guidance_hist:
                    mutual_cond = torch.cat([mutual_cond] * 3, dim=0)
                else:
                    mutual_cond = torch.cat([mutual_cond] * 2, dim=0)
            else:
                if do_classifier_free_guidance_mutual:
                    null_mutual_cond = torch.stack([null_latent] * mutual_cond.shape[0])
                    mutual_cond = torch.cat([mutual_cond, null_mutual_cond], dim=0)
                elif do_classifier_free_guidance_hist:
                    mutual_cond = torch.cat([mutual_cond] * 2, dim=0)
                else:
                    pass
            
            scaled_latent_model_input = (1 - self.args.eta) * scaled_latent_model_input + self.args.eta * mutual_cond
            scaled_latent_model_input = torch.cat([scaled_latent_model_input, hist_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                scaled_latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            if do_classifier_free_guidance:
                noise_pred_allcond, noise_pred_cate_mutual, noise_pred_cate, noise_pred_uncond = noise_pred.chunk(4)
                noise_pred = (
                    noise_pred_uncond
                    + hist_guidance_scale * (noise_pred_allcond - noise_pred_cate_mutual)
                    + mutual_guidance_scale * (noise_pred_cate_mutual - noise_pred_cate)
                    + category_guidance_scale * (noise_pred_cate - noise_pred_uncond)
                )
            elif do_classifier_free_guidance_category:
                if do_classifier_free_guidance_hist:
                    noise_pred_cate_hist, noise_pred_cate, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + hist_guidance_scale * (noise_pred_cate_hist - noise_pred_cate)
                        + category_guidance_scale * (noise_pred_cate - noise_pred_uncond)
                    )
                elif do_classifier_free_guidance_mutual:
                    noise_pred_cate_mutual, noise_pred_cate, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + mutual_guidance_scale * (noise_pred_cate_mutual - noise_pred_cate)
                        + category_guidance_scale * (noise_pred_cate - noise_pred_uncond)
                    )
                else:
                    noise_pred_cate, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + category_guidance_scale * (noise_pred_cate - noise_pred_uncond)
                    )
            else:
                if do_classifier_free_guidance_hist:
                    noise_pred_hist, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + hist_guidance_scale * (noise_pred_hist - noise_pred_uncond)
                    )
                elif do_classifier_free_guidance_mutual:
                    noise_pred_mutual, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + mutual_guidance_scale * (noise_pred_mutual - noise_pred_uncond)
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            prev_latents = latents.to(dtype=null_latent.dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, category_prompts.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            all_results = {}
            for i,uid in enumerate(fill_uids):
                uid = uid.item()
                oid = fill_oids[i].item()
                if uid not in all_results:
                    all_results[uid] = {}
                if oid not in all_results[uid]:
                    all_results[uid][oid] = {}
                    all_results[uid][oid]["images"] = []
                    all_results[uid][oid]["cates"] = []
                    all_results[uid][oid]["full_cates"] = full_cate[i]
                all_results[uid][oid]["images"].append(image[i])
                all_results[uid][oid]["cates"].append(fill_cate[i])
                all_results[uid][oid]["outfits"] = olists[fill_idx[i][0]]  
            
            return all_results, init_latents

        return (StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), fill_uids, fill_oids, fill_cate, full_cate, init_latents)
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_model_cpu_offload
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            # self.to("cpu", silence_dtype_warnings=True)
            self.to(torch.device("cpu"))
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

def ssim_postprocess(images, do_denormalize=True):

    def denormalize(images):
        return ((images / 2) + 0.5).clamp(0., 1.)

    images = denormalize(images) if do_denormalize else images
    images = images * 255

    return images

def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)

