import numpy as np

def generate_c2w(camera_radius, camera_position=None):
    if camera_position is None:
        # Generate random spherical coordinates for camera position (within the range mentioned in the DreamGaussian paper)
        deg_30_in_rad = np.deg2rad(30)
        phi = np.random.uniform(-np.pi, np.pi)  # Azimuthal angle 
        theta = np.random.uniform(-deg_30_in_rad, deg_30_in_rad)  # Polar angle

        # Convert spherical coordinates to cartesian
        camera_position = np.array([
            camera_radius * np.cos(theta) * np.sin(phi),
            camera_radius * np.sin(theta),
            camera_radius * np.cos(theta) * np.cos(phi)
        ])

    camera_look_at = np.array([0.0, 0.0, 0.0])
    camera_up = np.array([0.0, 1.0, 0.0])

    # Calculate c2w matrix
    z_axis = (camera_position - camera_look_at) / np.linalg.norm(camera_position - camera_look_at)
    x_axis = np.cross(camera_up, z_axis) / np.linalg.norm(np.cross(camera_up, z_axis))
    y_axis = np.cross(z_axis, x_axis)

    c2w = np.eye(4)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = z_axis
    c2w[:3, 3] = camera_position

    c2w = c2w.astype(np.float32)

    return c2w
