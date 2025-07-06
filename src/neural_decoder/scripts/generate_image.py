import os
import argparse
from pathlib import Path
import numpy as np
import torch
from diffusers import StableUnCLIPImg2ImgPipeline # type: ignore

from configs.config import config


def main():
    args = parse_args()
    sub_id = args.sub_id
    model_name = args.model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating images for subject {sub_id} with model {model_name}.")

    pred_dir = f"{config.data.predicted_embedding_dir}{model_name}/sub-{sub_id:02d}/"
    pred_clip_path = f"{pred_dir}pred_clip.npy"
    pred_vae_path = f"{pred_dir}pred_vae.npy"

    if not Path(pred_clip_path).exists() or not Path(pred_vae_path).exists():
        raise FileNotFoundError(f"Predicted embeddings not found for subject {sub_id} with model {model_name}.")

    pred_clip = np.load(pred_clip_path)
    pred_vae = np.load(pred_vae_path)

    print("Predicted embeddings loaded successfully.")

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    ).to(device)

    print("Diffusion pipeline loaded successfully.")

    recon_dir = f"results/unclip/{model_name}/sub-{sub_id:02d}/"
    os.makedirs(recon_dir, exist_ok=True)
    
    for i, embedding in enumerate(pred_clip):
        vae_latent = pred_vae[i].reshape((1, 4, 96, 96))
        vae_latent = torch.from_numpy(vae_latent).to(device).half()
        torch.manual_seed(0)
        noise_latent=torch.randn(vae_latent.shape, device=device).half()
        vae_latent = vae_latent*0.02 + noise_latent
        embedding = torch.tensor(embedding, device=device, dtype=torch.float16).unsqueeze(0)
        image = pipe(image_embeds=embedding, latents=vae_latent, guidance_scale=7.5).images[0] # type: ignore
        image.save(recon_dir + f"{i}.png")

    print("Images generated successfully.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_id", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True, choices=["ridge", "mlp"])
    return parser.parse_args()

if __name__ == "__main__":
    main()