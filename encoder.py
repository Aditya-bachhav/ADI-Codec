import numpy as np
import struct
import os
import cv2
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from sklearn.cluster import MiniBatchKMeans

# --- ADI v28 FAST CONFIG ---
VERSION = b'AD28'
VOCAB_SIZE = 1024
LIQUID_THRESH = 8.0

# --- VECTORIZED ENGINE ---
def create_grids():
    """Pre-compute the 8x8 coordinate grids for Liquid Math."""
    x = np.linspace(-3.5, 3.5, 8)
    y = np.linspace(-3.5, 3.5, 8)
    xv, yv = np.meshgrid(x, y)
    return xv.astype(np.float32), yv.astype(np.float32)

XV, YV = create_grids()
XV_FLAT = XV.flatten()
YV_FLAT = YV.flatten()
DENOM_X = np.sum(XV**2) # Pre-calc denominator for slope fitting
DENOM_Y = np.sum(YV**2)

# --- ENCODER (Vectorized) ---
def encode_v28_fast(source_file):
    out_dir = "outputs"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_file = os.path.join(out_dir, "image_v28_fast.adi")
    print(f"--- ADI v28 FAST ENCODER ---")
    
    start_time = time.time()
    
    # 1. LOAD
    img = Image.open(source_file).convert('RGB')
    w_orig, h_orig = img.size
    # Align to 8x8
    w, h = (w_orig // 8) * 8, (h_orig // 8) * 8
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    
    ycbcr = img.convert('YCbCr')
    y = np.array(ycbcr.split()[0], dtype=np.float32)
    cb = np.array(ycbcr.split()[1])
    cr = np.array(ycbcr.split()[2])

    # 2. BASE & RESIDUAL
    y_tiny = cv2.resize(y, (w//8, h//8), interpolation=cv2.INTER_AREA)
    cb_tiny = cv2.resize(cb, (w//8, h//8), interpolation=cv2.INTER_AREA)
    cr_tiny = cv2.resize(cr, (w//8, h//8), interpolation=cv2.INTER_AREA)
    y_pred = cv2.resize(y_tiny, (w, h), interpolation=cv2.INTER_CUBIC)
    residual = y - y_pred

    # 3. ANALYSIS (Vectorized)
    rows, cols = h // 8, w // 8
    # Shape: (N_Blocks, 64)
    blocks = residual.reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    
    # Mode Decision
    variances = np.var(blocks, axis=1)
    mode_map = (variances > LIQUID_THRESH) # True=Solid, False=Liquid
    
    n_solid = np.sum(mode_map)
    n_liquid = len(blocks) - n_solid
    print(f"Solid: {n_solid} | Liquid: {n_liquid}")
    
    # Initialize Output Arrays
    # 32-bit Packet: [Mode 1b] [Data 31b]
    stream = np.zeros(len(blocks), dtype=np.uint32)
    
    # --- PROCESS SOLIDS (VQ) ---
    if n_solid > 0:
        solid_blocks = blocks[mode_map]
        means_s = np.mean(solid_blocks, axis=1)
        stds_s = np.std(solid_blocks, axis=1)
        stds_s[stds_s < 0.1] = 1.0
        norm_s = (solid_blocks - means_s[:, None]) / stds_s[:, None]
        
        # Train Dictionary
        print("Learning Dictionary...")
        kmeans = MiniBatchKMeans(n_clusters=min(VOCAB_SIZE, len(solid_blocks)), batch_size=4096, n_init=1).fit(norm_s)
        codebook = kmeans.cluster_centers_.astype(np.float32)
        tokens = kmeans.predict(norm_s)
        
        # Pack Solids (Mode bit 1)
        # Structure: [1 (1b)] [Token (10b)] [Sigma (6b)] [Mu (8b)]
        p_mode = (1 << 31)
        p_tok = (tokens.astype(np.uint32) & 0x3FF) << 14
        p_sig = (np.clip(stds_s, 0, 63).astype(np.uint32)) << 8
        p_mu  = (np.clip(means_s + 128, 0, 255).astype(np.uint32))
        
        stream[mode_map] = p_mode | p_tok | p_sig | p_mu
    else:
        codebook = np.zeros((1, 64), dtype=np.float32)

    # --- PROCESS LIQUIDS (Math) ---
    if n_liquid > 0:
        liquid_mask = ~mode_map
        liquid_blocks = blocks[liquid_mask]
        
        # Vectorized Slope Fitting (Dot Product)
        # Slope = sum(blk * grid) / sum(grid^2)
        slopes_x = liquid_blocks @ XV_FLAT / DENOM_X
        slopes_y = liquid_blocks @ YV_FLAT / DENOM_Y
        means_l = np.mean(liquid_blocks, axis=1)
        
        # Pack Liquids (Mode bit 0)
        # Structure: [0 (1b)] [SlopeX (6b)] [SlopeY (6b)] [Mu (8b)]
        # Map slope -4.0..4.0 -> 0..63
        qsx = np.clip((slopes_x + 4.0) * 8.0, 0, 63).astype(np.uint32)
        qsy = np.clip((slopes_y + 4.0) * 8.0, 0, 63).astype(np.uint32)
        qmu = np.clip(means_l + 128, 0, 255).astype(np.uint32)
        
        p_sx = qsx << 14
        p_sy = qsy << 8
        
        stream[liquid_mask] = p_sx | p_sy | qmu

    # 4. WRITE FILE
    import lzma
    c_color = lzma.compress(y_tiny.astype(np.uint8).tobytes() + cb_tiny.tobytes() + cr_tiny.tobytes(), preset=5)
    c_book = lzma.compress(codebook.astype(np.float16).tobytes(), preset=5)
    c_inst = lzma.compress(stream.tobytes(), preset=5) # LZMA crushes the 32-bit padding
    
    with open(out_file, 'wb') as f:
        f.write(VERSION)
        f.write(struct.pack('<HH', w, h))
        f.write(struct.pack('<III', len(c_color), len(c_book), len(c_inst)))
        f.write(c_color); f.write(c_book); f.write(c_inst)

    orig_size = os.path.getsize(source_file) / 1024
    adi_size = os.path.getsize(out_file) / 1024
    print(f"Time: {time.time()-start_time:.2f}s | Original: {orig_size:.2f} KB | ADI: {adi_size:.2f} KB")
    return out_file

# --- DECODER (Vectorized) ---
def decode_v28_fast(path):
    try:
        import lzma
        with open(path, 'rb') as f:
            if f.read(4) != VERSION: return None, "Version Error"
            w, h = struct.unpack('<HH', f.read(4))
            sizes = struct.unpack('<III', f.read(12))
            blobs = [f.read(s) for s in sizes]
            
        # Base
        raw_color = lzma.decompress(blobs[0])
        pc = (w//8)*(h//8)
        y_t = np.frombuffer(raw_color[:pc], dtype=np.uint8).reshape(h//8, w//8)
        cb_t = np.frombuffer(raw_color[pc:pc*2], dtype=np.uint8).reshape(h//8, w//8)
        cr_t = np.frombuffer(raw_color[pc*2:], dtype=np.uint8).reshape(h//8, w//8)
        
        y_pred = cv2.resize(y_t, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        cb = cv2.resize(cb_t, (w, h), interpolation=cv2.INTER_CUBIC)
        cr = cv2.resize(cr_t, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Codebook
        codebook = np.frombuffer(lzma.decompress(blobs[1]), dtype=np.float16).astype(np.float32).reshape(-1, 64)
        
        # Stream (Load all as uint32)
        stream = np.frombuffer(lzma.decompress(blobs[2]), dtype=np.uint32)
        
        # --- VECTORIZED RECONSTRUCTION ---
        # Masks
        is_solid = (stream >> 31) & 1
        mask_solid = is_solid == 1
        mask_liquid = is_solid == 0
        
        # Initialize Detail Layer (N_Blocks, 64)
        detail_flat = np.zeros((len(stream), 64), dtype=np.float32)
        
        # Solids
        if np.any(mask_solid):
            s_data = stream[mask_solid]
            tok = (s_data >> 14) & 0x3FF
            sig = ((s_data >> 8) & 0x3F).astype(np.float32)
            mu  = (s_data & 0xFF).astype(np.float32) - 128.0
            
            # Fancy Indexing Lookup
            shapes = codebook[tok]
            detail_flat[mask_solid] = shapes * sig[:, None] + mu[:, None]
            
        # Liquids
        if np.any(mask_liquid):
            l_data = stream[mask_liquid]
            sx_raw = (l_data >> 14) & 0x3F
            sy_raw = (l_data >> 8) & 0x3F
            mu     = (l_data & 0xFF).astype(np.float32) - 128.0
            
            sx = (sx_raw.astype(np.float32) / 8.0) - 4.0
            sy = (sy_raw.astype(np.float32) / 8.0) - 4.0
            
            # Broadcasting: (N, 64) = (N,1)*XV + (N,1)*YV + (N,1)
            planes = (sx[:, None] * XV_FLAT) + (sy[:, None] * YV_FLAT) + mu[:, None]
            detail_flat[mask_liquid] = planes

        # Reshape to Image
        rows, cols = h // 8, w // 8
        detail_map = detail_flat.reshape(rows, cols, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)
        
        # Polish (Light deblock)
        y_raw = np.clip(y_pred + detail_map, 0, 255).astype(np.uint8)
        grid_mask = np.zeros_like(y_raw, dtype=np.uint8)
        grid_mask[:, 8:w:8] = 255; grid_mask[8:h:8, :] = 255
        blurred = cv2.GaussianBlur(y_raw, (3, 3), 0.5)
        y_raw[grid_mask == 255] = blurred[grid_mask == 255]
        
        return Image.merge('YCbCr', (
            Image.fromarray(y_raw, 'L'),
            Image.fromarray(cb, 'L'),
            Image.fromarray(cr, 'L')
        )).convert('RGB'), "Liquid Fast Active"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)

# --- GUI ---
class ADIFastLiquidJudge:
    def __init__(self, root):
        self.root = root; self.root.title("ADI Judge v28 (Fast Liquid)"); self.root.geometry("1400x800"); self.root.configure(bg="#000")
        self.img_s = None; self.img_a = None; self.zoom = 2.0
        f = tk.Frame(root, bg="#222"); f.pack(fill=tk.X)
        tk.Button(f, text="Load Source", command=self.load_s, bg="#444", fg="white").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(f, text="ENCODE v28 FAST", command=self.do_encode, bg="#0088ff", fg="white").pack(side=tk.LEFT, padx=10, pady=5)
        self.lbl = tk.Label(f, text="Ready", bg="#222", fg="gray"); self.lbl.pack(side=tk.LEFT, padx=20)
        vp = tk.Frame(root, bg="black"); vp.pack(fill=tk.BOTH, expand=True)
        self.cl = tk.Canvas(vp, bg="black"); self.cl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cr = tk.Canvas(vp, bg="black"); self.cr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cl.bind("<Motion>", self.mv); self.cr.bind("<Motion>", self.mv)

    def do_encode(self):
        if not hasattr(self, 'path_s'): return
        self.lbl.config(text="Vectorized Processing...", fg="cyan"); self.root.update()
        out = encode_v28_fast(self.path_s)
        img, msg = decode_v28_fast(out)
        self.img_a = img; self.lbl.config(text=f"Encoded: {msg}", fg="#00ff00"); self.upd()

    def load_s(self):
        p = filedialog.askopenfilename()
        if p: self.path_s = p; self.img_s = Image.open(p).convert("RGB"); self.upd()

    def upd(self):
        w, h = 680, 700
        if self.img_s: self.tk_s = ImageTk.PhotoImage(self.img_s.copy().resize((w,h))); self.cl.create_image(w//2, h//2, image=self.tk_s)
        if self.img_a: self.tk_a = ImageTk.PhotoImage(self.img_a.copy().resize((w,h))); self.cr.create_image(w//2, h//2, image=self.tk_a)

    def mv(self, e):
        if not self.img_s or not self.img_a: return
        r = 100
        ix = int(e.x * (self.img_s.width / self.cl.winfo_width())); iy = int(e.y * (self.img_s.height / self.cl.winfo_height()))
        cs = self.img_s.crop((ix-r, iy-r, ix+r, iy+r)).resize((int(r*2*self.zoom), int(r*2*self.zoom)))
        ca = self.img_a.crop((ix-r, iy-r, ix+r, iy+r)).resize((int(r*2*self.zoom), int(r*2*self.zoom)))
        self.tks = ImageTk.PhotoImage(cs); self.tka = ImageTk.PhotoImage(ca)
        self.cl.delete("Z"); self.cr.delete("Z")
        self.cl.create_image(e.x, e.y, image=self.tks, tags="Z"); self.cr.create_image(e.x, e.y, image=self.tka, tags="Z")

if __name__ == "__main__":
    root = tk.Tk(); ADIFastLiquidJudge(root); root.mainloop()