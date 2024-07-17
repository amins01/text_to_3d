import math
import torch
from gs.gaussian_model import GaussianModel

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from utils.sh_utils import eval_sh

class Renderer:
    def __init__(self, sh_degree=3, bg_color=[1, 1, 1]):
        self.gaussians = GaussianModel(sh_degree)
        self.bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        """From DreamGaussian code"""
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }