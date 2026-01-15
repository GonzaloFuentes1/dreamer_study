import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["EGL_PLATFORM"]="surfaceless"

# os.environ["MUJOCO_EGL_DEVICE_ID"] = "7"

import time
import numpy as np
from dm_control import suite
from tqdm import tqdm

# Intentar importar OpenCV para un resize rápido, fallback a Pillow si no existe
try:
    import cv2
    def resize_method(img, size):
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    RESIZE_LIB = "OpenCV"
except ImportError:
    from PIL import Image
    def resize_method(img, size):
        return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))
    RESIZE_LIB = "Pillow"

def bench_pixels(domain="walker", task="walk",
                 render_res=64, target_res=None,
                 camera_id=0, n_steps=2000, seed=0): # Reduced steps for quick check

    env = suite.load(domain_name=domain, task_name=task,
                     task_kwargs={"random": seed})
    ts = env.reset()

    spec = env.action_spec()
    rng = np.random.default_rng(seed)

    # Descripción de la tarea
    desc = f"Render {render_res}x{render_res}"
    if target_res and render_res != target_res:
        desc += f" -> Resize {target_res}x{target_res}"
    else:
        desc += " (Nativo)"

    # Warmup
    for _ in range(20):
        a = rng.uniform(spec.minimum, spec.maximum, size=spec.shape).astype(spec.dtype)
        ts = env.step(a)
        pixels = env.physics.render(height=render_res, width=render_res, camera_id=camera_id)
        if target_res and render_res != target_res:
            _ = resize_method(pixels, target_res)
        if ts.last():
            ts = env.reset()

    t0 = time.perf_counter()
    for _ in tqdm(range(n_steps), desc=desc):
        a = rng.uniform(spec.minimum, spec.maximum, size=spec.shape).astype(spec.dtype)
        ts = env.step(a)
        
        # 1. Render
        pixels = env.physics.render(height=render_res, width=render_res, camera_id=camera_id)
        
        # 2. Resize (si aplica)
        if target_res and render_res != target_res:
            pixels = resize_method(pixels, target_res)
            
        if ts.last():
            ts = env.reset()
    t1 = time.perf_counter()

    sps = n_steps / (t1 - t0)
    ms_per_step = 1000.0 / sps
    return sps, ms_per_step

print(f"\n--- Environment Info ---")
print(f"MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'Not Set')}")
print(f"EGL_PLATFORM: {os.environ.get('EGL_PLATFORM', 'Not Set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

print(f"\n--- Benchmarking Resize Strategies (Lib: {RESIZE_LIB}) ---")
print("Meta: Obtener observación de 64x64 para el Autoencoder\n")

# Caso 1: Render directo a 64x64 (Lo hace el motor gráfico)
sps1, ms1 = bench_pixels(render_res=64, target_res=64, n_steps=1000)
print(f"1. Render Directo 64x64:       {sps1:,.0f} SPS | {ms1:.3f} ms/step")
