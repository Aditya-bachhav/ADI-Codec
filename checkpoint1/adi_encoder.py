import numpy as np
import struct
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import os

# --- 1. CONFIGURATION ---
SOURCE_IMAGE = "source1.jpg"
OUTPUT_FILE = "prototype.adi"
ALPHABET_SIZE = 8192  # Your "Master Tier" vocab
BLOCK_SIZE = 4

if not os.path.exists(SOURCE_IMAGE):
    print(f"ERROR: {SOURCE_IMAGE} not found. Please place it in the folder.")
    exit()

# --- 2. THE DECONSTRUCTION (Translating Pixels) ---
print(f"Loading {SOURCE_IMAGE}...")
img = Image.open(SOURCE_IMAGE).convert('RGB').resize((3840, 2160))
data = np.asarray(img)
h, w, _ = data.shape

print("Slicing into 4x4 Clumps...")
blocks = data.reshape(h // BLOCK_SIZE, BLOCK_SIZE, w // BLOCK_SIZE, BLOCK_SIZE, 3)
blocks = blocks.swapaxes(1, 2).reshape(-1, BLOCK_SIZE * BLOCK_SIZE * 3)

# --- 3. THE INSIGHT (Pattern Recognition) ---
print(f"Generating {ALPHABET_SIZE}-word Alphabet (This is the slow part)...")
kmeans = MiniBatchKMeans(n_clusters=ALPHABET_SIZE, random_state=0, batch_size=4096, n_init="auto")
indices = kmeans.fit_predict(blocks).astype(np.uint16)
dictionary = kmeans.cluster_centers_.astype(np.uint8)

# --- 4. THE BINARY PACKER (Compressing like Hell) ---
print(f"Translating to Raw Binary: {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "wb") as f:
    # Header: Magic(4 bytes) + Width(2) + Height(2)
    f.write(struct.pack('4sHH', b'ADI1', w, h))
    
    # Write the Stream of Pointers (2 bytes each)
    for idx in indices:
        f.write(struct.pack('H', idx))

# Save the dictionary (The Alphabet) separately - in a real app, this is shared
np.save("adi_dictionary.npy", dictionary)

# --- 5. THE RESULT ---
original_size = os.path.getsize(SOURCE_IMAGE) / 1024
final_size = os.path.getsize(OUTPUT_FILE) / 1024
print(f"\n--- ENCODING SUCCESS ---")
print(f"Original JPG: {original_size:.2f} KB")
print(f"Final .adi:   {final_size:.2f} KB")
print(f"Compression Ratio: {original_size / final_size:.2f}x")