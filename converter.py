import numpy as np
import struct
import lzma
import cv2
import argparse
import os
from PIL import Image

# --- CONFIG ---
QUANT_SCALE = 5.0

def create_dct_matrix(n):
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == 0: matrix[i, j] = 1 / np.sqrt(n)
            else: matrix[i, j] = np.sqrt(2/n) * np.cos((2*j + 1) * i * np.pi / (2*n))
    return matrix

DCT_MAT = create_dct_matrix(8)
DCT_MAT_T = DCT_MAT.T
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32) * (QUANT_SCALE / 10.0)
Q_FLAT = Q_MATRIX.flatten()

def liquid_polish(img_y):
    # Gentle deblocking
    return cv2.GaussianBlur(img_y, (3, 3), 0.5)

def convert_adi(input_path, output_path):
    print(f"--- ADI CONVERTER ---")
    print(f"Reading: {input_path}")
    
    if not os.path.exists(input_path):
        print("Error: Input file not found.")
        return

    try:
        with open(input_path, 'rb') as f:
            if f.read(4) != b'AD10': 
                print("Error: Invalid ADI Version or File Format")
                return
            
            w, h = struct.unpack('<HH', f.read(4))
            sizes = struct.unpack('<IIIIII', f.read(24))
            blobs = [f.read(s) for s in sizes]
            
        # 1. BASE
        skin_raw = lzma.decompress(blobs[0])
        sy, sc = (h//4)*(w//4), (h//16)*(w//16)
        y_s = np.frombuffer(skin_raw[:sy], dtype=np.uint8).reshape(h//4, w//4)
        cb_s = np.frombuffer(skin_raw[sy:sy+sc], dtype=np.uint8).reshape(h//16, w//16)
        cr_s = np.frombuffer(skin_raw[sy+sc:], dtype=np.uint8).reshape(h//16, w//16)
        
        y_base = cv2.resize(y_s, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        cb = cv2.resize(cb_s, (w, h), interpolation=cv2.INTER_CUBIC)
        cr = cv2.resize(cr_s, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # 2. STREAMS
        mode_map = np.frombuffer(lzma.decompress(blobs[1]), dtype=np.uint8).reshape(h//8, w//8)
        smooth_coeffs = np.frombuffer(lzma.decompress(blobs[2]), dtype=np.int8).astype(np.float32)
        dictionary = np.frombuffer(lzma.decompress(blobs[3]), dtype=np.int8).astype(np.float32).reshape(-1, 64)
        tokens = np.frombuffer(lzma.decompress(blobs[4]), dtype=np.int16)
        gains_byte = np.frombuffer(lzma.decompress(blobs[5]), dtype=np.uint8)
        
        # 3. RECONSTRUCT
        detail = np.zeros((h, w), dtype=np.float32)
        rows, cols = h//8, w//8
        s_ptr, t_ptr = 0, 0
        
        for r in range(rows):
            for c in range(cols):
                if mode_map[r, c] == 0:
                    # ANALOG
                    q_blk = smooth_coeffs[s_ptr : s_ptr+64]
                    s_ptr += 64
                    dct_blk = (q_blk * Q_FLAT).reshape(8, 8)
                    detail[r*8:(r+1)*8, c*8:(c+1)*8] = DCT_MAT_T @ (dct_blk @ DCT_MAT)
                else:
                    # DIGITAL
                    tk = tokens[t_ptr]
                    g_val = gains_byte[t_ptr]
                    t_ptr += 1
                    
                    gain = 2.0 ** ((float(g_val) - 128.0) / 64.0)
                    dct_blk = (dictionary[tk] * gain).reshape(8, 8)
                    detail[r*8:(r+1)*8, c*8:(c+1)*8] = DCT_MAT_T @ (dct_blk @ DCT_MAT)
        
        y_rec = np.clip(y_base + detail, 0, 255).astype(np.uint8)
        y_rec = liquid_polish(y_rec)
        
        # 4. SAVE
        final_img = Image.merge('YCbCr', (
            Image.fromarray(y_rec, 'L'),
            Image.fromarray(cb, 'L'),
            Image.fromarray(cr, 'L')
        )).convert('RGB')
        
        final_img.save(output_path)
        print(f"Success: Saved to {output_path}")

    except Exception as e:
        print(f"Conversion Failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ADI to Image")
    parser.add_argument("input", help="Input .adi file")
    parser.add_argument("output", help="Output image path (.jpg, .png)")
    args = parser.parse_args()
    
    convert_adi(args.input, args.output)