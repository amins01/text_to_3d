import threading
from types import SimpleNamespace
from matplotlib import pyplot as plt
import numpy as np
import torch

from gs.renderer import Renderer
from utils.gs import generate_cuboid_pcd, generate_gs_360_animation, generate_gs_images, generate_sphere_pcd

class Trainer:
    def __init__(self, sd_model, prompt, num_splats, training_args, render_resolutions,
                 negative_prompt="", camera_radius=2.5, fovy=49, znear=0.01, zfar=100, sh_degree=0):
        # SD model
        self.sd_model = sd_model
        self.sd_model.generate_text_embeddings(prompt, negative_prompt)

        # Camera
        self.render_resolutions = render_resolutions
        camera_args = {
            "radius": camera_radius,
            "gs_image_width": render_resolutions[-1],
            "gs_image_height": render_resolutions[-1],
            "znear": znear,
            "zfar": zfar,
            "fovy": np.deg2rad(fovy),
            "fovx": 2 * np.arctan(np.tan(fovy / 2))
        }
        self.camera_args = SimpleNamespace(**camera_args)

        # Gaussian splats
        self.renderer = Renderer(sh_degree=sh_degree)
        
        # pcd = generate_cuboid_pcd(num_splats=num_splats, width=0.5, height=1.0, depth=0.5)
        pcd = generate_sphere_pcd(num_splats=num_splats, radius=0.5)
        self.renderer.gaussians.create_from_pcd(pcd)
        self.renderer.gaussians.training_setup(training_args)

        self.optimizer = self.renderer.gaussians.optimizer

    def _get_render_resolution(self, epoch, epochs):
        epoch_ratio = epoch / epochs
        num_res = len(self.render_resolutions)

        for i in range(1, num_res):
            if epoch_ratio < (i / num_res):
                return self.render_resolutions[i-1]

        return self.render_resolutions[-1]

    def train_step(self, epoch, epochs, batch_size):
        self.renderer.gaussians.update_learning_rate(epoch)

        # Set render resolution based on current epoch
        res = self._get_render_resolution(epoch, epochs)
        self.camera_args.gs_image_width = res
        self.camera_args.gs_image_height = res

        # Generate image(s) of the splats
        images = generate_gs_images(renderer=self.renderer, camera_args=self.camera_args, batch_size=batch_size)

        # if epoch % 20 == 0:
        #     plt.imshow(images[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
        #     plt.show()

        # Calculate loss
        loss = self.sd_model.train_step(images)

        # Backpropagate
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, batch_size, epochs=500, anim_camera_args=None):
        # Start real-time 360 animation in a separate thread
        if anim_camera_args is not None:
            animation_thread = threading.Thread(target=generate_gs_360_animation, 
                                                args=(self.renderer, anim_camera_args))
            animation_thread.start()
        
        for epoch in range(epochs):
            loss = self.train_step(epoch=epoch, epochs=epochs, batch_size=batch_size)
            print(f"Epoch {epoch + 1}, loss: {loss}")
