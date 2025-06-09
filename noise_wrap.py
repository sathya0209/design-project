import numpy as np

def estimate_flow(image):
    H, W = image.size
    return np.ones((H, W, 2)) * 2  # move 2px right

def warp_noise(shape, flow):
    H, W = shape
    noise = np.random.randn(H, W, 3)
    warped = np.roll(noise, shift=2, axis=1)  # simulate simple warp
    return warped
