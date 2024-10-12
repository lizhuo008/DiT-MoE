import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="/root/autodl-tmp/DiT-XL-2-dit_moe_xl_8E2A-size-256-cfg-1.5-seed-2024-singlegpu")
    parser.add_argument("--num_samples", type=int, default=5000)
    args = parser.parse_args()
    
    sample_dir = args.sample_dir
    num_samples = args.num_samples
    create_npz_from_sample_folder(sample_dir, num_samples)