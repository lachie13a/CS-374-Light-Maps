import numpy as np
from vedo import Mesh, Plotter, Sphere

def visualize_pro_lightmap(obj_path, lightmap_path, light_pos):
    mesh = Mesh(obj_path)
    
    mesh.texture(lightmap_path)
    
    # Lighting is already generated
    mesh.lighting('off')
    
    light_bulb = Sphere(pos=light_pos, r=0.1).color("yellow")
    plt = Plotter(axes=0, bg='blackboard')
    plt.background((0, 0, 0))
    plt.show(mesh, light_bulb, "Final OBJ Lightmap Result")

L_POS = [1.5, 1.5, 0]
visualize_pro_lightmap("Sphere.obj", "colored_lightmap.png", L_POS)