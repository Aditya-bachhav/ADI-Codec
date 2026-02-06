import numpy as np
from PIL import Image

# 1. Load our existing data
dictionary = np.load("adi_dictionary.npy")
alphabet_indices = np.load("adi_stream.npy")
ghost_data = np.asarray(Image.open("adi_ghost_detail.png")).astype(np.int16) - 128

# 2. Extract the 'i' Token (Intensity Correction)
# We calculate the average error for each 4x4 clump to 'nudge' the light
h, w, _ = ghost_data.shape
block_size = 4
# Reshape ghost into blocks and find the mean intensity correction
i_tokens = ghost_data.reshape(h // block_size, block_size, w // block_size, block_size, 3)
i_tokens = i_tokens.mean(axis=(1, 3)).astype(np.int8) # This is our tiny 'i' token

# 3. Save the NEW Master Stream
# Now we have Alphabet Indices AND Intensity Corrections
np.save("adi_i_layer.npy", i_tokens)

print("--- MASTER QUALITY SECURED ---")
print("The 'i' tokens have been extracted. Total extra data: very small.")