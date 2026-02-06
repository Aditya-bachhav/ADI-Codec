import numpy as np
import struct
from PIL import Image
import os

# --- 1. CONFIGURATION ---
INPUT_FILE = "prototype.adi"
DICTIONARY_FILE = "adi_dictionary.npy"
OUTPUT_IMAGE = "adi_result_binary.jpg"
BLOCK_SIZE = 4

if not os.path.exists(INPUT_FILE) or not os.path.exists(DICTIONARY_FILE):
    print("ERROR: .adi file or dictionary missing. Run the Encoder first.")
    exit()

# --- 2. UNPACKING THE BITSTREAM ---
print(f"Opening {INPUT_FILE}...")
with open(INPUT_FILE, "rb") as f:
    # Read Header: Magic(4), Width(2), Height(2)
    header = f.read(8)
    magic, width, height = struct.unpack('4sHH', header)
    
    if magic != b'ADI1':
        print("ERROR: Not a valid .adi file.")
        exit()
    
    # Read the payload (The Alphabet Pointers)
    raw_payload = f.read()
    # Each 'H' is a 2-byte unsigned short (16-bit index)
    indices = struct.unpack(f'{len(raw_payload)//2}H', raw_payload)

# --- 3. THE RECONSTRUCTION (Translating Back) ---
print("Loading Alphabet...")
dictionary = np.load(DICTIONARY_FILE)

grid_h, grid_w = height // BLOCK_SIZE, width // BLOCK_SIZE

print(f"Rebuilding {width}x{height} image from {len(indices)} instructions...")
# Map indices back to the pixel clumps
reconstructed_blocks = dictionary[list(indices)].reshape(grid_h, grid_w, BLOCK_SIZE, BLOCK_SIZE, 3)

# Swap axes to put pixels back in spatial order
final_arr = reconstructed_blocks.swapaxes(1, 2).reshape(height, width, 3)

# --- 4. THE RESULT ---
final_img = Image.fromarray(final_arr.astype(np.uint8))
# We save as high-quality JPG for the final visual comparison
final_img.save(OUTPUT_IMAGE, "JPEG", quality=95)

print(f"\n--- DECODING SUCCESS ---")
print(f"Result saved to: {OUTPUT_IMAGE}")