import numpy as np
import struct
from PIL import Image
import zlib
import os

INPUT_FILE = "outputs/smart_adaptive.adi"
OUTPUT_IMAGE = "outputs/adaptive_result.jpg"

print("=== ADAPTIVE DECODER LOADING ===")

with open(INPUT_FILE, 'rb') as f:
    magic = f.read(4)
    if magic != b'ADIA': exit("Not an Adaptive .adi file")
    
    # 1. Read Header
    w, h, _, _, block_size = struct.unpack('<HHHHB', f.read(9))
    print(f"Dimensions: {w}x{h} | Block Size: {block_size}")
    
    # 2. Extract Codebook A (Smooth)
    len_smooth = struct.unpack('<I', f.read(4))[0]
    dict_smooth = np.frombuffer(zlib.decompress(f.read(len_smooth)), dtype=np.uint8)
    dict_smooth = dict_smooth.reshape(256, block_size, block_size, 3)
    
    # 3. Extract Codebook B (Sharp)
    len_sharp = struct.unpack('<I', f.read(4))[0]
    dict_sharp = np.frombuffer(zlib.decompress(f.read(len_sharp)), dtype=np.uint8)
    dict_sharp = dict_sharp.reshape(512, block_size, block_size, 3)
    
    # 4. Extract Indices and Types
    len_indices = struct.unpack('<I', f.read(4))[0]
    indices = np.frombuffer(zlib.decompress(f.read(len_indices)), dtype=np.uint16)
    
    len_types = struct.unpack('<I', f.read(4))[0]
    types_packed = np.frombuffer(zlib.decompress(f.read(len_types)), dtype=np.uint8)
    types = np.unpackbits(types_packed)[:len(indices)] # Unpack bits back to 0/1 array

print("Codebooks loaded. Reconstructing...")

# 5. THE PROJECTION (Switching Alphabets)
# We reconstruct block by block based on the 'Type' bit
grid_h, grid_w = h // block_size, w // block_size
final_image = np.zeros((h, w, 3), dtype=np.uint8)

idx_counter = 0
for i in range(grid_h):
    for j in range(grid_w):
        idx = indices[idx_counter]
        type_bit = types[idx_counter]
        
        # THE SWITCH: If bit is 0 use Smooth Dict, if 1 use Sharp Dict
        if type_bit == 0:
            block_data = dict_smooth[idx]
        else:
            block_data = dict_sharp[idx]
            
        # Paint the block
        y_start, x_start = i * block_size, j * block_size
        final_image[y_start:y_start+block_size, x_start:x_start+block_size] = block_data
        
        idx_counter += 1

# 6. Save
Image.fromarray(final_image).save(OUTPUT_IMAGE, 'JPEG', quality=95)
print(f"Success! Check {OUTPUT_IMAGE}")