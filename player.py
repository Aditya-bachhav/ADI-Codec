import numpy as np
import struct
import lzma
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# --- ADI v20 JUDGE CONFIG ---
QUANT_SCALE = 4.5

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

def diamond_polish(img_y, mode_map):
    """
    v20 DIAMOND POLISH ENGINE:
    1. Deblock ONLY the seams (Grid lines).
    2. Sharpen ONLY the texture blocks (Contrast Pop).
    """
    h, w = img_y.shape
    
    # 1. SURGICAL DEBLOCKING (Grid Only)
    # Mask 8x8 grid
    grid_mask = np.zeros_like(img_y, dtype=np.uint8)
    grid_mask[:, 0:w:8] = 255 
    grid_mask[0:h:8, :] = 255
    
    # Blur
    seam_blur = cv2.GaussianBlur(img_y, (3, 3), 0)
    
    # Apply blur ONLY to grid lines
    polished = img_y.copy()
    polished[grid_mask == 255] = seam_blur[grid_mask == 255]
    
    # 2. TEXTURE POP (Unsharp Mask)
    # We only sharpen where mode_map == 1 (Texture Blocks)
    # Resize mode_map (blocks) -> pixel mask
    texture_mask_blocks = (mode_map == 1).astype(np.uint8) * 255
    # Nearest neighbor resize to keep block boundaries hard
    texture_mask_pixels = cv2.resize(texture_mask_blocks, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create sharpened version
    gaussian = cv2.GaussianBlur(polished, (0, 0), 2.0)
    # Original + (Original - Blur) * Amount
    sharpened = cv2.addWeighted(polished, 1.5, gaussian, -0.5, 0)
    
    # Apply sharp pixels ONLY inside texture blocks
    final_y = polished.copy()
    final_y[texture_mask_pixels == 255] = sharpened[texture_mask_pixels == 255]
    
    return np.clip(final_y, 0, 255).astype(np.uint8)

def decode_adi_v20(path):
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'AD20': return None, f"Invalid Version: {magic}"
            
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
        
        # 3. RECONSTRUCT (Dual Core)
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
                    # DIGITAL (Log Gain)
                    tk = tokens[t_ptr]
                    g_val = gains_byte[t_ptr]
                    t_ptr += 1
                    
                    gain = 2.0 ** ((float(g_val) - 128.0) / 64.0)
                    dct_blk = (dictionary[tk] * gain).reshape(8, 8)
                    detail[r*8:(r+1)*8, c*8:(c+1)*8] = DCT_MAT_T @ (dct_blk @ DCT_MAT)
        
        y_rec = np.clip(y_base + detail, 0, 255).astype(np.uint8)
        
        # 4. DIAMOND POLISH (Using mode_map)
        y_rec = diamond_polish(y_rec, mode_map)
        
        return Image.merge('YCbCr', (
            Image.fromarray(y_rec, 'L'),
            Image.fromarray(cb, 'L'),
            Image.fromarray(cr, 'L')
        )).convert('RGB'), "Success"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

# --- JUDGE GUI ---
class ADIJudgeV20:
    def __init__(self, root):
        self.root = root
        self.root.title("ADI Judge v20.0 (The Diamond)")
        self.root.geometry("1400x800")
        self.root.configure(bg="#000")
        self.img_s, self.img_a = None, None
        self.zoom = 3.0
        
        f = tk.Frame(root, bg="#111")
        f.pack(fill=tk.X)
        tk.Button(f, text="Load JPG (Source)", command=self.load_s, bg="#333", fg="white").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(f, text="Load v20 ADI", command=self.load_a, bg="#00ffaa", fg="black").pack(side=tk.LEFT, padx=10, pady=5)
        self.lbl = tk.Label(f, text="Ready", bg="#111", fg="gray")
        self.lbl.pack(side=tk.LEFT, padx=20)

        vp = tk.Frame(root, bg="black"); vp.pack(fill=tk.BOTH, expand=True)
        self.cl = tk.Canvas(vp, bg="black", cursor="crosshair", highlightthickness=0); self.cl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cr = tk.Canvas(vp, bg="black", cursor="crosshair", highlightthickness=0); self.cr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.cl.bind("<Motion>", self.mv); self.cr.bind("<Motion>", self.mv)
        root.bind("<MouseWheel>", self.wh)

    def load_s(self):
        # Auto-load source.jpg if present, else ask
        if os.path.exists("source.jpg"):
            self.img_s = Image.open("source.jpg").convert("RGB")
            self.upd()
            self.lbl.config(text="Loaded source.jpg automatically", fg="white")
        else:
            p = filedialog.askopenfilename()
            if p: self.img_s = Image.open(p).convert("RGB"); self.upd()

    def load_a(self):
        p = filedialog.askopenfilename(initialdir="outputs")
        if p:
            img, s = decode_adi_v20(p)
            if img: self.img_a = img; self.lbl.config(text=f"Loaded v20 | Size: {img.size}", fg="lime"); self.upd()
            else: self.lbl.config(text=s, fg="red")

    def upd(self):
        w, h = 680, 700
        if self.img_s:
            self.tk_s = ImageTk.PhotoImage(self.img_s.copy().resize((w,h)))
            self.cl.create_image(w//2, h//2, image=self.tk_s)
        if self.img_a:
            self.tk_a = ImageTk.PhotoImage(self.img_a.copy().resize((w,h)))
            self.cr.create_image(w//2, h//2, image=self.tk_a)

    def mv(self, e):
        if not self.img_s or not self.img_a: return
        dw, dh = self.tk_s.width(), self.tk_s.height()
        ox, oy = (self.cl.winfo_width()-dw)//2, (self.cl.winfo_height()-dh)//2
        ix = int((e.x - ox) * (self.img_s.width / dw))
        iy = int((e.y - oy) * (self.img_s.height / dh))
        
        r, z = 120, self.zoom
        ix = max(r, min(self.img_s.width-r, ix))
        iy = max(r, min(self.img_s.height-r, iy))
        
        cs = self.img_s.crop((ix-r, iy-r, ix+r, iy+r)).resize((int(r*2*z), int(r*2*z)), 0)
        ca = self.img_a.crop((ix-r, iy-r, ix+r, iy+r)).resize((int(r*2*z), int(r*2*z)), 0)
        
        self.tks = ImageTk.PhotoImage(cs); self.tka = ImageTk.PhotoImage(ca)
        self.cl.delete("L"); self.cr.delete("L")
        cx, cy = self.cl.winfo_width()//2, self.cl.winfo_height()//2
        
        self.cl.create_image(cx, cy, image=self.tks, tags="L")
        self.cl.create_text(cx, cy-r*z-20, text="SOURCE", fill="lime", font=("Arial", 14, "bold"), tags="L")
        self.cr.create_image(cx, cy, image=self.tka, tags="L")
        self.cr.create_text(cx, cy-r*z-20, text="ADI v20 (Diamond)", fill="#00ffaa", font=("Arial", 14, "bold"), tags="L")

    def wh(self, e): self.zoom = max(1.0, self.zoom + (0.5 if e.delta > 0 else -0.5))

if __name__ == "__main__":
    root = tk.Tk()
    app = ADIJudgeV20(root)
    root.mainloop()