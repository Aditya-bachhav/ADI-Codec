import numpy as np
from PIL import Image

# 1. Load the two versions
original = Image.open("source.jpg").convert('RGB').resize((3840, 2160))
reconstructed = Image.open("adi_reconstructed.jpg").convert('RGB')

# 2. Convert to Arrays
orig_arr = np.asarray(original).astype(np.int16)
recon_arr = np.asarray(reconstructed).astype(np.int16)

# 3. Calculate the "Ghost" (The Residuals)
# We subtract the alphabet version from the original
ghost = orig_arr - recon_arr

# 4. Normalize the Ghost for visibility
# We add 128 so 'zero difference' looks grey, not black
visible_ghost = np.clip(ghost + 128, 0, 255).astype(np.uint8)

# 5. Save the Ghost
Image.fromarray(visible_ghost).save("adi_ghost_detail.png")

print("--- GHOST ANALYSIS COMPLETE ---")
print("Open 'adi_ghost_detail.png'. Everything you see there is what the 'i' token needs to carry.")