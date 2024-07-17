import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_model_key):
        super().__init__()
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(sd_model_key, torch_dtype=torch.float16).to(device)
        self.scheduler = DDIMScheduler.from_pretrained(sd_model_key, subfolder="scheduler")
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    def encode_images(self, images):
        images = 2 * images - 1 # Normalize to [-1, 1]
        return self.pipe.vae.encode(images).latent_dist.sample() * self.pipe.vae.config.scaling_factor

    def encode_text(self, prompt):
        inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.pipe.text_encoder(inputs.input_ids.to(self.device))[0]

    def generate_text_embeddings(self, prompts, negative_prompts):
        self.pos_prompt_embedding = self.encode_text(prompts)
        self.neg_prompt_embedding = self.encode_text(negative_prompts)

    def train_step(self, images, guidance_scale=100):
        images = torch.cat(images, dim=0).to(torch.float16)
        batch_size = images.shape[0]

        # Generate image latents
        latents = self.encode_images(images)

        # Generate random timestep(s)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), dtype=torch.long, device=self.device)

        # Generate random gaussian noise
        noise = torch.randn_like(latents)

        # Apply generated noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        # Weighting function
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # Duplicate for CFG
        noisy_latents = torch.cat([noisy_latents] * 2)
        t = torch.cat([t] * 2)

        # Expand the positive and negative prompts to match batch size and concatenate them
        embeddings = torch.cat([self.pos_prompt_embedding.expand(batch_size, -1, -1), self.neg_prompt_embedding.expand(batch_size, -1, -1)])

        with torch.no_grad():
            # Predict noise
            noise_pred = self.pipe.unet(noisy_latents, t, encoder_hidden_states=embeddings).sample

        # Guidance (to increase prompt influence)
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        added_noise = noisy_latents - latents
        grad = w * (noise_pred - added_noise)
        target = (latents - grad).detach()

        loss = F.mse_loss(latents.float(), target, reduction='mean')
        # loss = F.mse_loss(noise_pred, added_noise, reduction='mean')

        return loss

