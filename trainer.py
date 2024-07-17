import threading
from types import SimpleNamespace
from matplotlib import pyplot as plt
import numpy as np
import torch

from gs.renderer import Renderer
from utils.gs import generate_cuboid_pcd, generate_gs_360_animation, generate_gs_images, generate_sphere_pcd

class Trainer:
    def __init__(self, sd_model, prompt, num_splats, training_args, gs_image_width, gs_image_height,
                 anim_360=True, negative_prompt="", camera_radius=2.5, fovy=49, camera_width=800,
                 camera_height=800, sh_degree=0, znear=0.01, zfar=100):
        # SD model
        self.sd_model = sd_model
        self.sd_model.generate_text_embeddings(prompt, negative_prompt)

        # Camera
        camera_args = {
            "radius": camera_radius,
            "gs_image_width": gs_image_width,
            "gs_image_height": gs_image_height,
            "znear": znear,
            "zfar": zfar,
            "fovy": np.deg2rad(fovy),
            "fovx": 2 * np.arctan(np.tan(fovy / 2) * camera_width / camera_height)
        }
        self.camera_args = SimpleNamespace(**camera_args)

        # Gaussian splats
        self.renderer = Renderer(sh_degree=sh_degree)
        self.anim_360 = anim_360
        
        # pcd = generate_cuboid_pcd(num_splats=num_splats, width=0.5, height=1.0, depth=0.5)
        pcd = generate_sphere_pcd(num_splats=num_splats, radius=0.5)
        self.renderer.gaussians.create_from_pcd(pcd)
        self.renderer.gaussians.training_setup(training_args)

        self.optimizer = self.renderer.gaussians.optimizer

    def train_step(self, epoch, batch_size):
        self.renderer.gaussians.update_learning_rate(epoch)

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

    def train(self, batch_size, epochs=500):
        # Start real-time 360 animation in a separate thread
        if self.anim_360:
            self.animation_thread = threading.Thread(target=generate_gs_360_animation, 
                                                     args=(self.renderer, self.camera_args, np.array([0, 0, self.camera_args.radius])))
            self.animation_thread.start()
        
        for epoch in range(epochs):
            loss = self.train_step(epoch=epoch, batch_size=batch_size)
            print(f"Epoch {epoch + 1}, loss: {loss}")

