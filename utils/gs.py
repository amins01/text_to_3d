import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gs.basic_point_cloud import BasicPointCloud
from gs.gaussian_model import MiniCam
from utils.sh_utils import SH2RGB
from utils.three_d import generate_c2w

def generate_cuboid_pcd(num_splats, width, height, depth):
    points = []
    colors = []
    normals = []

    for _ in range(num_splats):
        # Generate random points within the cuboid's volume
        x = np.random.uniform(-width / 2, width / 2)
        y = np.random.uniform(-height / 2, height / 2)
        z = np.random.uniform(-depth / 2, depth / 2)
        point = [x, y, z]
        sh_color = np.random.random((3,)) / 255.0
        normal = [0, 0, 0]

        points.append(point)
        colors.append(sh_color)
        normals.append(normal)

    points = np.array(points)
    colors = np.array(colors)
    normals = np.array(normals)

    return BasicPointCloud(
        points=points, colors=SH2RGB(colors), normals=normals
    )

def generate_sphere_pcd(num_splats, radius):
    points = []
    colors = []
    normals = []

    for _ in range(num_splats):
        # Generate points within the sphere's volume
        phi = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        r = radius * np.random.uniform(0, 1)**(1/3) # Uniform distribution within sphere
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        point = [x, y, z]
        sh_color = np.random.random((3,)) / 255.0
        normal = [0, 0, 0]

        points.append(point)
        colors.append(sh_color)
        normals.append(normal)

    points = np.array(points)
    colors = np.array(colors)
    normals = np.array(normals)

    return BasicPointCloud(
        points=points, colors=SH2RGB(colors), normals=normals
    )

def generate_gs_images(renderer, camera_args, batch_size=1, ready_to_display=False):
    rendered_images = []

    for _ in range(batch_size):
        # Generate random c2w matrix
        c2w = generate_c2w(camera_args.radius)

        rendered_output = renderer.render(
            viewpoint_camera=MiniCam(
                c2w,
                camera_args.gs_image_width,
                camera_args.gs_image_height,
                camera_args.fovy,
                camera_args.fovx,
                camera_args.znear,
                camera_args.zfar
            ),
            bg_color=torch.tensor(
                data=[1, 1, 1] if random.uniform(0, 1) < 0.5 else [0, 0, 0],
                dtype=torch.float32,
                device="cuda"
            )
        )

        if ready_to_display:
            rendered_image = rendered_output["image"].permute(1, 2, 0).cpu().detach().numpy()
        else:
            rendered_image = rendered_output["image"].unsqueeze(0)
        
        rendered_images.append(rendered_image)

    return rendered_images

def generate_gs_360_animation(renderer, camera_args, num_frames=360, fps=120, output_filename=None):
    '''Generates a 360-degree animation of Gaussian splats'''
    fig, ax = plt.subplots()
    
    initial_c2w = generate_c2w(camera_args.radius, camera_args.position)

    camera = MiniCam(
        initial_c2w,
        camera_args.gs_image_width,
        camera_args.gs_image_height,
        camera_args.fovy,
        camera_args.fovx,
        camera_args.znear,
        camera_args.zfar
    )

    # Display first frame
    image_tensor = renderer.render(camera)["image"].clamp(0, 1)
    image = ax.imshow(image_tensor.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')

    def update_frame(frame):
        theta = frame * (2 * np.pi / num_frames)

        # Rotation matrix around the y-axis
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        # Apply rotation to c2w matrix
        camera.c2w = np.dot(rotation_matrix, initial_c2w) 

        # Recalculate world_view_transform
        w2c = np.linalg.inv(camera.c2w)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1
        camera.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()

        # Convert to float32
        camera.world_view_transform = camera.world_view_transform.to(torch.float32)
        # Recalculate full_proj_transform 
        camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix

        # Render frame
        image_tensor = renderer.render(camera)["image"].clamp(0, 1)
        image.set_array(image_tensor.permute(1, 2, 0).cpu().detach().numpy())
        return image,

    ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, blit=True, interval=1000/fps)

    if output_filename:
        ani.save(f'{output_filename}.gif', fps=fps)

    plt.show()

