import numpy as np
import struct
from PIL import Image
import zlib
import os

INPUT_FILE = "outputs/smart_adaptive.adi"
OUTPUT_IMAGE = "outputs/adaptive_8x8_result.jpg"

print("=== ADI SMART READER ===")

if not os.path.exists(INPUT_FILE):
    exit("Error: Run the Encoder first!")

with open(INPUT_FILE, 'rb') as f:
    magic = f.read(4)
    if magic != b'ADIX': exit("Error: Format mismatch.")
    
    w, h, n_smooth, n_sharp, block_size = struct.unpack('<HHHHB', f.read(9))
    l_smooth, l_sharp, l_idx, l_type = struct.unpack('<IIII', f.read(16))
    
    print(f"Loading {w}x{h} image (Grid: {block_size}x{block_size})...")
    
    dict_smooth = np.frombuffer(zlib.decompress(f.read(l_smooth)), dtype=np.uint8)
    dict_smooth = dict_smooth.reshape(n_smooth, block_size, block_size, 3)
    
    dict_sharp = np.frombuffer(zlib.decompress(f.read(l_sharp)), dtype=np.uint8)
    dict_sharp = dict_sharp.reshape(n_sharp, block_size, block_size, 3)
    
    indices = np.frombuffer(zlib.decompress(f.read(l_idx)), dtype=np.uint16)
    types_packed = np.frombuffer(zlib.decompress(f.read(l_type)), dtype=np.uint8)
    types = np.unpackbits(types_packed)[:len(indices)]

print("Reconstructing...")
grid_h, grid_w = h // block_size, w // block_size
final_image = np.zeros((h, w, 3), dtype=np.uint8)

idx_counter = 0
for i in range(grid_h):
    for j in range(grid_w):
        idx = indices[idx_counter]
        block = dict_smooth[idx] if types[idx_counter] == 0 else dict_sharp[idx]
        y, x = i * block_size, j * block_size
        final_image[y:y+block_size, x:x+block_size] = block
        idx_counter += 1

Image.fromarray(final_image).save(OUTPUT_IMAGE, 'JPEG', quality=95)
print(f"Done: {OUTPUT_IMAGE}")