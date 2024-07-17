import torch
from types import SimpleNamespace

from models.stable_diffusion import StableDiffusion
from trainer import Trainer

if __name__ == "__main__":
    gs_image_width = 512
    gs_image_height = 512
    epochs = 500
    batch_size = 1
    num_splats = 5000

    training_args = {
        "num_pts": num_splats,
        "sh_degree": 0,
        "position_lr_init": 0.001,
        "position_lr_final": 0.00002,
        "position_lr_delay_mult": 0.02,
        "position_lr_max_steps": 300,
        "feature_lr": 0.01,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.005,
        "percent_dense": 0.01,
        "density_start_iter": 0,
        "density_end_iter": 3000,
        "densification_interval": 50,
        "opacity_reset_interval": 700,
        "densify_grad_threshold": 0.01
    }

    prompt = input("Enter your prompt: ")

    sd_model_key = "stabilityai/stable-diffusion-2-1-base"

    sd_model = StableDiffusion(device=torch.device("cuda"), sd_model_key=sd_model_key)

    trainer = Trainer(
        sd_model=sd_model,
        prompt=prompt,
        num_splats=num_splats,
        training_args=SimpleNamespace(**training_args),
        gs_image_width=gs_image_width,
        gs_image_height=gs_image_height
    )

    trainer.train(batch_size, epochs)