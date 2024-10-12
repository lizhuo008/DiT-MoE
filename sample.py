# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT-MoE.
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from diffusion.rectified_flow import RectifiedFlow 
from tqdm import tqdm
import argparse



def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.image_size in [256, 512]
    assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8 

    if args.model == "DiT-XL/2" or args.model == "DiT-G/2": 
        pretraining_tp=1
        use_flash_attn=True 
        dtype = torch.float16
    else:
        pretraining_tp=2
        use_flash_attn=False 
        dtype = torch.float32

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_experts=args.num_experts, 
        num_experts_per_tok=args.num_experts_per_tok,
        pretraining_tp=pretraining_tp,
        use_flash_attn=use_flash_attn
    ).to(device)

    if dtype == torch.float16:
        model = model.half()
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    
    if args.ckpt is None: 
        print('only for testing middle ckpts')
        if args.model == "DiT-S/2":
            ckpt_path = "dit_moe_s_8E2A.pt" 
        elif args.model == "DiT-B/2":
            ckpt_path = "dit_moe_b_8E2A.pt" 
        elif args.model == "DiT-XL/2": 
            ckpt_path = "results/deepspeed-DiT-XL-2-rf/checkpoints/tmp.pt" 
        else: 
            ckpt_path = "results/deepspeed-DiT-G-2-rf/checkpoints/tmp.pt" 
    else:
        ckpt_path = args.ckpt 


    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important! 

    if args.rf:
        diffusion = RectifiedFlow(model)
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device) 
    
    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.seed}-singlegpu"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")
    
    iterations = args.num_fid_samples
    pbar = range(iterations)
    pbar = tqdm(pbar) 
    
    numbers = torch.arange(args.num_classes)
    numbers_repeated = numbers.repeat_interleave(args.num_fid_samples // args.num_classes)
    numbers_rank = numbers_repeated.to(device)
    n = 1
    for i in pbar:
        # Create sampling noise:
        # n = len(class_labels)
        class_labels = numbers_rank[i * n: (i + 1) * n]
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        if dtype == torch.float16: 
            if args.rf: 
                with torch.autocast(device_type='cuda'):
                    STEPSIZE = 50
                    init_noise = torch.randn(n, 4, latent_size, latent_size, device=device) 
                    conds = torch.tensor(class_labels, device=device)
                    images = diffusion.sample_with_xps(init_noise, conds, null_cond = torch.tensor([1000] * n).cuda(), sample_steps = STEPSIZE, cfg = 1.5)
                    samples = vae.decode(images[-1] / 0.18215).sample
            
            else:
                with torch.autocast(device_type='cuda'):
                    samples = diffusion.p_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples / 0.18215).sample
        
        else:
            samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

        
        # Save and display images: 
        if args.model == "DiT-S/2":
            save_image(samples, "sample_s.png", nrow=4, normalize=True, value_range=(-1, 1))
        elif args.model == "DiT-B/2":
            save_image(samples, "sample_b.png", nrow=4, normalize=True, value_range=(-1, 1))
        elif args.model == "DiT-XL/2":
            save_image(samples, f"{sample_folder_dir}/{i:06d}.png", nrow=4, normalize=True, value_range=(-1, 1)) 
        else:
            save_image(samples, "sample_g.png", nrow=4, normalize=True, value_range=(-1, 1)) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-G/2")
    parser.add_argument("--vae-path", type=str, default="/maindata/data/shared/multimodal/zhengcong.fei/ckpts/sd-vae-ft-mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument('--num_experts', default=16, type=int,) 
    parser.add_argument('--num_experts_per_tok', default=2, type=int,) 
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-fid-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ckpt", type=str, default=None, )
    parser.add_argument("--rf", action='store_true')
    args = parser.parse_args()
    main(args) 
