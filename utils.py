from PIL import Image
import imageio

def load_image(path):
    return Image.open(path).convert("RGB").resize((256, 256))

def read_prompt(path):
    with open(path, "r") as f:
        return f.read().strip()

def save_gif(frames, path, fps=5):
    imageio.mimsave(path, frames, fps=fps)

def save_image(image, path):
    image.save(path)
