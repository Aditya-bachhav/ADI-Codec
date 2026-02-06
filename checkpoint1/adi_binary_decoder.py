import numpy as np
import struct
from PIL import Image

# 1. READ THE RAW BINARY (.adi)
print("Unpacking the .adi bitstream...")
with open("prototype.adi", "rb") as f:
    # Read the Header
    magic, width, height = struct.unpack('4sHH', f.read(8))
    # Read the Indices (16-bit unsigned integers)
    raw_data = f.read()
    indices = struct.unpack(f'{len(raw_data)//2}H', raw_data)

# 2. LOAD THE ALPHABET (The Dictionary)
dictionary = np.load("adi_dictionary.npy")

# 3. THE RECONSTRUCTION (The 'Translation' back to pixels)
block_size = 4
grid_h, grid_w = height // block_size, width // block_size

print(f"Rebuilding {width}x{height} image from binary...")
reconstructed_blocks = dictionary[list(indices)].reshape(grid_h, grid_w, block_size, block_size, 3)
final_arr = reconstructed_blocks.swapaxes(1, 2).reshape(height, width, 3)

# 4. SHOW THE RESULT
final_img = Image.fromarray(final_arr.astype(np.uint8))
final_img.save("binary_result.jpg", "JPEG", quality=95)
print("--- DECODE SUCCESS: Open 'binary_result.jpg' ---")