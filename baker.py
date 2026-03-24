import trimesh
import numpy as np
from PIL import Image
from tqdm import tqdm
from vedo import Mesh, Plotter, Sphere

def generate_colored_lightmap(obj_path, texture_size=1024):
    print(f"Loading mesh: {obj_path}...")
    # Trimesh loads the .obj and looks for the .mtl automatically
    mesh = trimesh.load(obj_path)
    
    # If the OBJ has multiple parts, merge them into one geometry
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    uvs = mesh.visual.uv
    faces = mesh.faces
    vertices = mesh.vertices
    normals = mesh.vertex_normals

    # Initialize a blank RGB image (Width, Height, 3 Channels)
    lightmap = np.zeros((texture_size, texture_size, 3), dtype=np.float128)

    # Create a light object
    light_pos = np.array([1.5, 1.5, 0])
    light_color = np.array([1.0, 0.829, 0.529]) 
    light_intensity = 1000.0

    ambient_color = np.array([0, 0, 0])

    print("Processing triangles and calculating light...")

    for face_idx, face in tqdm(enumerate(faces), total=len(faces), leave=False):
        # Get the 3 vertices and UVs for this specific triangle
        tri_v = vertices[face]   
        tri_uv = uvs[face]       
        tri_n = normals[face]    

        # Scale UVs (0 to 1) to actual pixel coordinates (0 to 1024)
        pixel_uv = tri_uv * (texture_size - 1)
        
        # Define a bounding box around the triangle to save processing time
        min_u, min_v = np.floor(pixel_uv.min(axis=0)).astype(int)
        max_u, max_v = np.ceil(pixel_uv.max(axis=0)).astype(int)

        # Stay within image bounds
        min_u, max_u = np.clip([min_u, max_u], 0, texture_size - 1)
        min_v, max_v = np.clip([min_v, max_v], 0, texture_size - 1)

        for v in range(min_v, max_v + 1):
            for u in range(min_u, max_u + 1):
                # Determining if pixel (u, v) is inside the triangle
                p = np.array([u, v])
                v0, v1, v2 = pixel_uv
                
                # Math to find weights (w, s, t)
                denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                
                # Skip any triangles
                if abs(denom) < 1e-6: continue
                
                w = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
                s = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
                t = 1 - w - s

                # If all weights are 0-1, the pixel is inside the triangle
                if w >= 0 and s >= 0 and t >= 0:
                    # Calculate exact 3D position and Normal for this pixel
                    world_pos = w*tri_v[0] + s*tri_v[1] + t*tri_v[2]
                    world_normal = w*tri_n[0] + s*tri_n[1] + t*tri_n[2]
                    
                    # Normalize the interpolated normal
                    norm_len = np.linalg.norm(world_normal)
                    if norm_len > 0:
                        world_normal /= norm_len

                    # Distance and direction to light
                    light_vec = light_pos - world_pos
                    distance = np.linalg.norm(light_vec)
                    light_dir = light_vec / (distance + 1e-6)

                    # Dot Product
                    dot = np.maximum(np.dot(world_normal, light_dir), 0.0)

                    # Attenuation (Light falloff over distance)
                    attenuation = light_intensity / (distance**2 + 1.0)
                    
                    # (Direct Light * Color) + Ambient "Fill" Light
                    direct_light = (dot * attenuation) * light_color
                    pixel_final_rgb = direct_light + ambient_color
                    
                    # Store in our lightmap array
                    lightmap[v, u] = pixel_final_rgb

    # Tone mapping / Normalization: Scale everything so the brightest pixel is 255
    max_val = lightmap.max()
    if max_val > 0:
        lightmap = (lightmap / max_val) * 255
    
    final_image = Image.fromarray(lightmap.astype(np.uint8), 'RGB')
    
    # Flip vertically because image coordinates (top-down) often differ from UV coordinates (bottom-up)
    final_image = final_image.transpose(Image.FLIP_TOP_BOTTOM)
    
    final_image.save("colored_lightmap.png")
    print("Success! Lightmap saved as 'colored_lightmap.png'")

generate_colored_lightmap("Sphere.obj")