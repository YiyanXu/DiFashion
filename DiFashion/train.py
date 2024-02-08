"""Script to fine-tune Stable Diffusion for Fashion Outfit Generation and Recommendation"""

import argparse
import logging
import math
import os

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import data_utils
from models.difashion import DiFashion, MutualEncoder

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0")

logger = get_logger(__name__, log_level="INFO")

def parse_all_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-base",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='/data/path/',
        help="A folder containing the dataset for training and inference."
    )
    parser.add_argument(
        '--img_folder_path',
        type=str,
        default='/data/path/xxx'
    )
    parser.add_argument(
        "--data_processed",
        type=bool,
        default=True,
        help="if the data is processed or not."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='',
        help="The name of the Dataset for training and inference."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/output/path/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/",
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_mutual_guidance",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--use_history",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.2,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--coupling_dropout_prob",
        type=float,
        default=0.3,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--cate_conditioning_dropout_prob",
        type=float,
        default=0.2,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--category_emb_size",
        type=int,
        default=64,
        help="Fashion item category embedding size.",
    )
    parser.add_argument(
        "--hid_dim",
        type=int,
        default=256,
        help="Fashion encoder hidden dim."
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="The weight of mutual guidance."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--category_guidance_scale",
        type=float,
        default=12.0
    )
    parser.add_argument(
        "--hist_guidance_scale",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--mutual_guidance_scale",
        type=float,
        default=5.0
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--use_ema_fashion", action="store_true", help="Whether to use EMA model for fashion encoder.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="fashion_outfit_generation",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--run_name", type=str, default='', help="Run name")  

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    args.output_dir = os.path.join(args.output_dir, args.run_name)
 
    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_all_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    
    if args.report_to == "wandb":
        if is_wandb_available():
            import wandb
            wandb.init(project="difashion")
        else:
            args.report_to = "tensorboard"

    logging_dir = args.logging_dir

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    device = accelerator.device

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Data loading......")
    data_path = os.path.join(args.data_path, args.dataset_name)

    if args.data_processed:
        train_dict = np.load(os.path.join(data_path, "processed", "new_train.npy"), allow_pickle=True).item()
        valid_fitb_dict = np.load(os.path.join(data_path, "processed", "new_fitb_valid.npy"), allow_pickle=True).item()
        train_history = np.load(os.path.join(data_path, "processed", "train_hist_latents.npy"), allow_pickle=True).item()
        valid_history = np.load(os.path.join(data_path, "processed", "valid_hist_latents.npy"), allow_pickle=True).item()
    else:
        train_dict = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
        valid_fitb_dict = np.load(os.path.join(data_path, "fitb_valid.npy"), allow_pickle=True).item()
        test_fitb_dict = np.load(os.path.join(data_path, "fitb_test.npy"), allow_pickle=True).item()

        train_history = np.load(os.path.join(data_path, "train_history.npy"), allow_pickle=True).item()
        valid_history = np.load(os.path.join(data_path, "valid_history.npy"), allow_pickle=True).item()
        test_history = np.load(os.path.join(data_path, "test_history.npy"), allow_pickle=True).item()

    valid_grd_dict = np.load(os.path.join(data_path, "valid_grd.npy"), allow_pickle=True).item()
    new_id_cate_dict = np.load(os.path.join(data_path, "new_id_cate_dict.npy"), allow_pickle=True).item()
    all_image_paths = np.load(os.path.join(data_path, "new_all_item_image_paths.npy"), allow_pickle=True)

    img_trans = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ]
    )
    img_dataset = data_utils.ImagePathDataset(args.img_folder_path, all_image_paths, img_trans, do_normalize=True)
    null_img = img_dataset[0].to(device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Build the diffusion model......")
    diffusion = DiFashion(args, logger, len(new_id_cate_dict), device)
    logger.info("Completed.")

    with accelerator.main_process_first():
        if args.data_processed:
            train_data_dict = train_dict
            valid_data_dict = valid_fitb_dict
            train_hist_latents = train_history
            valid_hist_latents = valid_history

            logger.info(f"Successfully loaded the processed data for training and validation.")
        else:
            logger.info(f"Preprocess datasets for DiFashion.")
            train_data_dict, train_hist_latents = data_utils.preprocess_dataset(train_dict, data_path,
                new_id_cate_dict, train_history, img_dataset, diffusion.tokenizer, diffusion.vae, device)

            save_path = os.path.join(data_path, "processed")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, "new_train.npy"), np.array(train_data_dict))
            np.save(os.path.join(save_path, "train_hist_latents.npy"), np.array(train_hist_latents))

            valid_data_dict, valid_hist_latents = data_utils.preprocess_dataset(valid_fitb_dict, data_path,
                new_id_cate_dict, valid_history, img_dataset, diffusion.tokenizer, diffusion.vae, device)
            
            np.save(os.path.join(save_path, "new_fitb_valid.npy"), np.array(valid_data_dict))
            np.save(os.path.join(save_path, "valid_hist_latents.npy"), np.array(valid_hist_latents))

            test_data_dict, test_hist_latents = data_utils.preprocess_dataset(test_fitb_dict, data_path,
                new_id_cate_dict, test_history, img_dataset, diffusion.tokenizer, diffusion.vae, device)
            
            np.save(os.path.join(save_path, "new_fitb_test.npy"), np.array(test_data_dict))
            np.save(os.path.join(save_path, "test_hist_latents.npy"), np.array(test_hist_latents))

            logger.info(f"Successfully processed and saved the dataset for training, validation and test into {save_path}.")

    train_dataset = data_utils.FashionDiffusionData(train_data_dict)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    valid_dataset = data_utils.FashionDiffusionData(valid_data_dict)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        shuffle=False, 
        batch_size=10,
        num_workers=args.dataloader_num_workers,
    )
    logger.info("dataloader built.")

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(diffusion.unet.parameters(), model_cls=UNet2DConditionModel, model_config=diffusion.unet.config)
    
    if args.use_ema_fashion:
        ema_encoder = EMAModel(diffusion.fashion_encoder.parameters(), model_cls=MutualEncoder, model_config=diffusion.fashion_encoder.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
            if args.use_ema_fashion:
                ema_encoder.save_pretrained(os.path.join(output_dir, "fashion_encoder_ema"))

            for i, model in enumerate(models):
                model.fashion_encoder.save_pretrained(os.path.join(output_dir, "fashion_encoder"))
                model.unet.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(device)
                del load_model
            
            if args.use_ema_fashion:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "fashion_encoder_ema"), MutualEncoder)
                ema_encoder.load_state_dict(load_model.state_dict())
                ema_encoder.to(device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.unet.register_to_config(**load_model.config)
                model.unet.load_state_dict(load_model.state_dict())
                del load_model

                # load mutual encoder into model
                load_model = MutualEncoder.from_pretrained(input_dir, subfolder="fashion_encoder")
                model.fashion_encoder.register_to_config(**load_model.config)
                model.fashion_encoder.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        diffusion.unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    logger.info("build the optimizer...")
    train_params = list(diffusion.unet.parameters()) + list(diffusion.fashion_encoder.parameters())
    optimizer = optimizer_cls(
        train_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    logger.info("Prepare everything with our accelerator...")
    diffusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        diffusion, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(device)
    
    if args.use_ema_fashion:
        ema_encoder.to(device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("difashion", config=tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        diffusion.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            mask_ratio = args.conditioning_dropout_prob
            coupling_mask_ratio = args.coupling_dropout_prob
            cate_mask_ratio = args.cate_conditioning_dropout_prob

            with accelerator.accumulate(diffusion):
                loss = diffusion(batch, img_dataset, train_hist_latents, null_img, mask_ratio, coupling_mask_ratio, cate_mask_ratio, weight_dtype, generator)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(diffusion.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(diffusion.unet.parameters())
                if args.use_ema_fashion:
                    ema_encoder.step(diffusion.fashion_encoder.parameters())

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            torch.cuda.empty_cache()
            
            # You can use the following codes to sampling some example images during training.

            # if accelerator.is_main_process:
            #     if global_step % 100 == 0:
            #         diffusion.eval()
            #         logger.info("Running validation...")
            #         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            #         unwrapped_model = accelerator.unwrap_model(diffusion)
            #         if args.use_ema:
            #             # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            #             ema_unet.store(diffusion.unet.parameters())
            #             ema_unet.copy_to(diffusion.unet.parameters())
            #         if args.use_ema_fashion:
            #             ema_encoder.store(diffusion.fashion_encoder.parameters())
            #             ema_encoder.copy_to(diffusion.fashion_encoder.parameters())

            #         with torch.autocast(
            #             str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            #         ):
            #             image_path = os.path.join("./", "samples")
            #             if not os.path.exists(image_path):
            #                 os.makedirs(image_path)
                        
            #             outputs = {}
            #             for i,batch in enumerate(valid_dataloader):
            #                 uids = batch["uids"].to(device)
            #                 oids = batch["oids"].to(device)
            #                 input_ids = batch["input_ids"].to(device)
            #                 category = batch["category"].to(device)
            #                 olists = batch["outfits"].to(device)
            #                 outfit_images = []
            #                 for olist in olists:
            #                     for iid in olist:
            #                         outfit_images.append(img_dataset[iid])
            #                 outfit_images = torch.stack(outfit_images).to(device)

            #                 batch_outputs, _ = unwrapped_model.fashion_generation(
            #                     uids,
            #                     oids,
            #                     input_ids,
            #                     olists,
            #                     outfit_images,
            #                     category,
            #                     valid_hist_latents,
            #                     num_inference_steps=args.num_inference_steps,
            #                     category_guidance_scale=args.category_guidance_scale,
            #                     hist_guidance_scale=args.hist_guidance_scale,
            #                     mutual_guidance_scale=args.mutual_guidance_scale,
            #                     null_img=null_img,
            #                     generator=generator,
            #                     return_dict=False
            #                 )
            #                 outputs.update(batch_outputs)
            #                 # if i > args.valid_batch_num:
            #                 if i > 3:
            #                     break

            #         for i,uid in enumerate(outputs):
            #             uid_image_path = os.path.join(image_path, str(uid))
            #             if not os.path.exists(uid_image_path):
            #                 os.makedirs(uid_image_path)
                        
            #             for oid in outputs[uid]:
            #                 images = outputs[uid][oid]["images"]
            #                 cates = outputs[uid][oid]["cates"]
            #                 full_cates = outputs[uid][oid]["full_cates"]

            #                 oid_image_path = os.path.join(uid_image_path, str(oid))
            #                 if not os.path.exists(oid_image_path):
            #                     os.makedirs(oid_image_path)

            #                 grd_exist = False
            #                 files_in_oid_image_path = os.listdir(oid_image_path)
            #                 for filename in files_in_oid_image_path:
            #                     if "grd" in filename:
            #                         grd_exist = True

            #                 if not grd_exist:
            #                     semantic_cates = []
            #                     for j,cate in enumerate(full_cates):
            #                         semantic_cates.append(new_id_cate_dict[cate.item()])

            #                     grd_imgs = []
            #                     for iid in valid_grd_dict[oid]["outfits"]:
            #                         img = Image.open(os.path.join(args.img_folder_path, all_image_paths[iid]))
            #                         grd_imgs.append(img)

            #                     merge_and_save_images(
            #                         grd_imgs,
            #                         os.path.join(oid_image_path, 'grd_' + '_'.join(semantic_cates) + '.jpg')
            #                     )
                            
            #                 semantic_cates = []
            #                 for j,cate in enumerate(cates):
            #                     semantic_cates.append(new_id_cate_dict[cate.item()])
            #                 merge_and_save_images(
            #                     images,
            #                     os.path.join(oid_image_path, f"{global_step}_{args.mutual_guidance_scale}_{args.hist_guidance_scale}_" + '_'.join(semantic_cates) + '.jpg')
            #                 )
                    
            #         if args.use_ema:
            #             # Switch back to the original UNet parameters.
            #             ema_unet.restore(diffusion.unet.parameters())
            #         if args.use_ema_fashion:
            #             ema_encoder.restore(diffusion.fashion_encoder.parameters())

            #         torch.cuda.empty_cache()

            if global_step >= 20000:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

def merge_and_save_images(images, save_path):
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
    
    merged_image.save(save_path)

if __name__ == "__main__":
    main()


