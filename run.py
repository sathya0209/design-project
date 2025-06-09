from utils import load_image, read_prompt, save_gif, save_image
from noise_wrap import estimate_flow, warp_noise
from diffusion import generate_video, replace_object_with_dummy

# Load input image and prompt
image = load_image("input/image.jpg")
prompt = read_prompt("input/prompt.txt")

# Ask user what they want to do
print("\nWhat would you like to do?")
print("1. Create animated GIF (move object)")
print("2. Remove object")
print("3. Replace object (with placeholder)")

choice = input("Enter 1, 2, or 3: ").strip()

# Common preprocessing
flow = estimate_flow(image)
warped_noise = warp_noise(image.size, flow)

# Settings
num_frames = 25
fps = 5

if choice == "1":
    # Move object and create GIF
    frames = generate_video(image, prompt, warped_noise, num_frames=num_frames, remove_object=False)
    save_gif(frames, "output/result.gif", fps=fps)
    print("✅ GIF created at output/result.gif")

elif choice == "2":
    # Remove object
    frames = generate_video(image, prompt, warped_noise, num_frames=1, remove_object=True)
    save_image(frames[0], "output/removed_object.png")
    print("✅ Object removed image saved at output/removed_object.png")

elif choice == "3":
    # Replace object (with dummy image — real AI model could be used later)
    replaced = replace_object_with_dummy(image)
    save_image(replaced, "output/replaced_object.png")
    print("✅ Object replaced image saved at output/replaced_object.png")

else:
    print("❌ Invalid input. Please enter 1, 2, or 3.")
