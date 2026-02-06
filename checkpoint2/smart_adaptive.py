import numpy as np
import struct
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import os
import zlib

SOURCE_IMAGE = "source.jpg"
OUTPUT_FILE = "outputs/smart_adaptive.adi"
DETAIL_THRESHOLD = 30 

# 1. LOAD & SECTOR ANALYSIS (Same as before)
img = Image.open(SOURCE_IMAGE).convert('RGB').resize((3840, 2160))
data = np.asarray(img)
h, w, _ = data.shape
blocks = data.reshape(h // 12, 12, w // 12, 12, 3)
sector_chaos = np.std(blocks, axis=(1, 3, 4))

# 2. GENERATE DUAL ALPHABETS
print("building dual codebooks...")
smooth_blocks = blocks[sector_chaos < DETAIL_THRESHOLD].reshape(-1, 12*12*3)
sharp_blocks = blocks[sector_chaos >= DETAIL_THRESHOLD].reshape(-1, 12*12*3)

# Codebook A: Smooth (256 Patterns)
kmeans_smooth = MiniBatchKMeans(n_clusters=256, n_init=3).fit(smooth_blocks)
dict_smooth = kmeans_smooth.cluster_centers_.astype(np.uint8)

# Codebook B: Sharp (512 Patterns)
kmeans_sharp = MiniBatchKMeans(n_clusters=512, n_init=3).fit(sharp_blocks)
dict_sharp = kmeans_sharp.cluster_centers_.astype(np.uint8)

# 3. ENCODE STREAM (Map indices to specific alphabets)
indices = []
types = [] 
print("tokenizing...")
for i in range(h // 12):
    for j in range(w // 12):
        block = blocks[i, :, j, :, :].reshape(1, -1)
        if sector_chaos[i, j] < DETAIL_THRESHOLD:
            indices.append(kmeans_smooth.predict(block)[0])
            types.append(0) # 0 = Smooth
        else:
            indices.append(kmeans_sharp.predict(block)[0])
            types.append(1) # 1 = Sharp

# 4. PACKAGING (The "Codebook" Transfer)
# We compress each part separately
blob_smooth = zlib.compress(dict_smooth.tobytes())
blob_sharp = zlib.compress(dict_sharp.tobytes())
blob_indices = zlib.compress(np.array(indices, dtype=np.uint16).tobytes())
blob_types = zlib.compress(np.packbits(types).tobytes())

print(f"Writing .adi file...")
with open(OUTPUT_FILE, 'wb') as f:
    f.write(b'ADIA') # Magic
    f.write(struct.pack('<HHHHB', w, h, 0, 0, 12)) # Header
    
    # We MUST write the lengths so the decoder knows how to slice the file!
    f.write(struct.pack('<I', len(blob_smooth)))
    f.write(blob_smooth)
    
    f.write(struct.pack('<I', len(blob_sharp)))
    f.write(blob_sharp)
    
    f.write(struct.pack('<I', len(blob_indices)))
    f.write(blob_indices)
    
    f.write(struct.pack('<I', len(blob_types)))
    f.write(blob_types)

print(f"--- SUCCESS ---")
print(f"Total Size: {os.path.getsize(OUTPUT_FILE)/1024:.2f} KB")