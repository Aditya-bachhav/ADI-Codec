import numpy as np
import struct
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import os
import zlib

# --- CONFIGURATION ---
SOURCE_IMAGE = "source.jpg"
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "smart_adaptive.adi")

BLOCK_SIZE = 12
DETAIL_THRESHOLD = 25  # Lower = more blocks get 'Sharp' treatment (better quality, slightly larger file)

# 1. LOAD AND PREP
print(f"Loading {SOURCE_IMAGE}...")
if not os.path.exists(SOURCE_IMAGE):
    print("ERROR: source.jpg not found!")
    exit()

img = Image.open(SOURCE_IMAGE).convert('RGB')
# Resize to 4K for the prototype standard
img = img.resize((3840, 2160)) 
data = np.asarray(img)
h, w, _ = data.shape

# 2. SECTOR ANALYSIS (The "VarDCT" Logic)
# We calculate the standard deviation (chaos) for every 12x12 block
print("Analyzing sector chaos...")
blocks = data.reshape(h // BLOCK_SIZE, BLOCK_SIZE, w // BLOCK_SIZE, BLOCK_SIZE, 3)
# Axis=(1,3,4) calculates std dev across the pixels in each block
sector_chaos = np.std(blocks, axis=(1, 3, 4)) 

# 3. GENERATE DUAL ALPHABETS
print("Generating Dual Codebooks (This takes a moment)...")

# Split blocks into two piles: Smooth (Sky/Water) and Sharp (Trees/Edges)
flat_shape = (-1, BLOCK_SIZE * BLOCK_SIZE * 3)
smooth_training_data = blocks[sector_chaos < DETAIL_THRESHOLD].reshape(flat_shape)
sharp_training_data = blocks[sector_chaos >= DETAIL_THRESHOLD].reshape(flat_shape)

print(f"  - Smooth Blocks: {len(smooth_training_data)}")
print(f"  - Sharp Blocks:  {len(sharp_training_data)}")

# Codebook A: Smooth (256 Patterns - Standard Byte)
kmeans_smooth = MiniBatchKMeans(n_clusters=256, n_init=3, batch_size=2048).fit(smooth_training_data)
dict_smooth = kmeans_smooth.cluster_centers_.astype(np.uint8)

# Codebook B: Sharp (512 Patterns - High Fidelity)
kmeans_sharp = MiniBatchKMeans(n_clusters=512, n_init=3, batch_size=2048).fit(sharp_training_data)
dict_sharp = kmeans_sharp.cluster_centers_.astype(np.uint8)

# 4. ENCODE THE STREAM (The "Map")
print("Tokenizing image stream...")
indices = []
types = [] # 0 = Smooth, 1 = Sharp

grid_h = h // BLOCK_SIZE
grid_w = w // BLOCK_SIZE

# We iterate through the grid and assign the best 'letter' from the correct alphabet
for i in range(grid_h):
    for j in range(grid_w):
        # Extract single block
        block_pixels = blocks[i, :, j, :, :].reshape(1, -1)
        
        # Check Chaos Level
        if sector_chaos[i, j] < DETAIL_THRESHOLD:
            # Use Smooth Alphabet
            idx = kmeans_smooth.predict(block_pixels)[0]
            indices.append(idx)
            types.append(0)
        else:
            # Use Sharp Alphabet
            idx = kmeans_sharp.predict(block_pixels)[0]
            indices.append(idx)
            types.append(1)

# 5. BINARY PACKING (The Protocol)
print("Packing binary data...")

# A. Convert Lists to Arrays
indices_arr = np.array(indices, dtype=np.uint16)
types_arr = np.array(types, dtype=np.uint8) 

# B. Compress Each Blob
# We use ZLIB (Deflate) to crush the data size
blob_smooth = zlib.compress(dict_smooth.tobytes(), level=9)
blob_sharp = zlib.compress(dict_sharp.tobytes(), level=9)
blob_indices = zlib.compress(indices_arr.tobytes(), level=9)
# packbits turns [0,1,1,0,0,0,0,0] into a single byte - massive savings
blob_types = zlib.compress(np.packbits(types_arr).tobytes(), level=9)

# 6. WRITE THE FILE (With Length Headers)
with open(OUTPUT_FILE, 'wb') as f:
    # --- GLOBAL HEADER (13 bytes) ---
    f.write(b'ADIA')  # Magic: ADI Adaptive
    f.write(struct.pack('<HHHHB', w, h, 0, 0, BLOCK_SIZE)) 

    # --- LENGTH HEADERS (Crucial for Decoder) ---
    # We write 4 integers (4 bytes each) telling the decoder the size of each blob
    f.write(struct.pack('<IIII', 
        len(blob_smooth), 
        len(blob_sharp), 
        len(blob_indices), 
        len(blob_types)
    ))

    # --- THE PAYLOAD ---
    f.write(blob_smooth)
    f.write(blob_sharp)
    f.write(blob_indices)
    f.write(blob_types)

# 7. REPORT
size_kb = os.path.getsize(OUTPUT_FILE) / 1024
print(f"\n--- ENCODING COMPLETE ---")
print(f"File Saved: {OUTPUT_FILE}")
print(f"Total Size: {size_kb:.2f} KB")
print(f"Visual Protocol: {grid_w}x{grid_h} grid ({len(indices)} instructions)")