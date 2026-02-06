import numpy as np
import struct
import os
import cv2
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.cluster import MiniBatchKMeans

# --- ADI v30 FLUX CONFIG ---
VERSION = b'ADV1'
VOCAB_SIZE = 1024
BLOCK_SIZE = 8
LIQUID_THRESH = 10.0
MOTION_THRESH = 30.0  # Threshold to decide if a block has "Moved" (Pixel diff)
GOP_SIZE = 30         # Group of Pictures: 1 I-Frame every 30 frames

# --- VECTORIZED MATH UTILS ---
def create_grids():
    x = np.linspace(-3.5, 3.5, 8); y = np.linspace(-3.5, 3.5, 8)
    xv, yv = np.meshgrid(x, y)
    return xv.astype(np.float32).flatten(), yv.astype(np.float32).flatten()
XV, YV = create_grids()
DENOM_X, DENOM_Y = np.sum(XV**2), np.sum(YV**2)

# --- ENCODER ENGINE ---
class FluxEncoder:
    def __init__(self):
        self.codebook = None
        
    def encode_frame(self, frame_y, prev_y, is_iframe):
        h, w = frame_y.shape
        rows, cols = h // 8, w // 8
        
        # 1. Base Layer (Tiny)
        y_tiny = cv2.resize(frame_y, (w//8, h//8), interpolation=cv2.INTER_AREA)
        y_pred = cv2.resize(y_tiny, (w, h), interpolation=cv2.INTER_CUBIC)
        residual = frame_y - y_pred
        
        # 2. Block Analysis
        blocks = residual.reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        
        # 3. Motion Detection (Skip Logic)
        if not is_iframe and prev_y is not None:
            # Compare current frame to previous frame
            # We compare the RAW pixels to see if visual change occurred
            diff = np.abs(frame_y - prev_y)
            diff_blocks = diff.reshape(rows, 8, cols, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
            block_diffs = np.sum(diff_blocks, axis=1)
            # Mask: 0 = Skip (Copy Prev), 1 = Update
            update_mask = (block_diffs > MOTION_THRESH)
        else:
            # I-Frame updates everything
            update_mask = np.ones(len(blocks), dtype=bool)
            
        active_blocks = blocks[update_mask]
        
        # 4. Mode Decision (Liquid vs Solid)
        # Only analyze active blocks
        stream_chunk = np.zeros(len(active_blocks), dtype=np.uint32)
        
        if len(active_blocks) > 0:
            variances = np.var(active_blocks, axis=1)
            is_solid = variances > LIQUID_THRESH
            
            # --- SOLIDS (Texture) ---
            if np.any(is_solid):
                solids = active_blocks[is_solid]
                means = np.mean(solids, axis=1)
                stds = np.std(solids, axis=1); stds[stds < 0.1] = 1.0
                norm = (solids - means[:, None]) / stds[:, None]
                
                # I-Frame: Learn Dictionary
                # P-Frame: Reuse Dictionary
                if is_iframe or self.codebook is None:
                    kmeans = MiniBatchKMeans(n_clusters=min(VOCAB_SIZE, len(norm)), batch_size=4096, n_init=1).fit(norm)
                    self.codebook = kmeans.cluster_centers_.astype(np.float32)
                    # Pad codebook if small
                    if len(self.codebook) < VOCAB_SIZE:
                        pad = np.zeros((VOCAB_SIZE - len(self.codebook), 64), dtype=np.float32)
                        self.codebook = np.vstack([self.codebook, pad])
                
                # Match tokens (using existing codebook)
                # Dot product for speed
                scores = np.dot(norm, self.codebook.T)
                tokens = np.argmax(scores, axis=1)
                
                # Pack: [1 (1b)] [Tok (10b)] [Sig (6b)] [Mu (8b)]
                p_tok = (tokens.astype(np.uint32) & 0x3FF) << 14
                p_sig = (np.clip(stds, 0, 63).astype(np.uint32)) << 8
                p_mu  = (np.clip(means + 128, 0, 255).astype(np.uint32))
                stream_chunk[is_solid] = (1 << 31) | p_tok | p_sig | p_mu

            # --- LIQUIDS (Math) ---
            if np.any(~is_solid):
                liquids = active_blocks[~is_solid]
                sx = liquids @ XV / DENOM_X
                sy = liquids @ YV / DENOM_Y
                mu = np.mean(liquids, axis=1)
                
                # Pack: [0 (1b)] [Sx (6b)] [Sy (6b)] [Mu (8b)]
                qsx = np.clip((sx + 4.0) * 8.0, 0, 63).astype(np.uint32) << 14
                qsy = np.clip((sy + 4.0) * 8.0, 0, 63).astype(np.uint32) << 8
                qmu = np.clip(mu + 128, 0, 255).astype(np.uint32)
                stream_chunk[~is_solid] = qsx | qsy | qmu
                
        return y_tiny, update_mask, stream_chunk

def encode_video(source_path):
    out_dir = "outputs"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_file = os.path.join(out_dir, "video_v30.adv")
    
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened(): return None
    
    w_raw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_raw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Align to 8x8
    w, h = (w_raw // 8) * 8, (h_raw // 8) * 8
    
    print(f"--- ADI v30 FLUX VIDEO ---")
    print(f"Source: {w}x{h} @ {fps}fps ({count} frames)")
    
    encoder = FluxEncoder()
    frames_data = []
    
    import lzma
    
    prev_y = None
    start_time = time.time()
    
    # We will process a max of 90 frames for this demo (to prevent memory overflow in python)
    max_frames = min(count, 90)
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # Resize & Convert
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # OpenCV uses YCrCb
        y = ycbcr[:,:,0].astype(np.float32)
        
        # Color channels (Subsampled once per frame)
        cr = cv2.resize(ycbcr[:,:,1], (w//8, h//8), interpolation=cv2.INTER_AREA)
        cb = cv2.resize(ycbcr[:,:,2], (w//8, h//8), interpolation=cv2.INTER_AREA)
        
        is_iframe = (i % GOP_SIZE == 0)
        
        # ENCODE CORE
        y_tiny, mask, stream = encoder.encode_frame(y, prev_y, is_iframe)
        
        # Reconstruct current frame (Simulate Decoder State for next P-Frame)
        # Note: In a real encoder, we'd do full decode loop here. 
        # For speed, we just update prev_y with source (Lossy drift risk, but okay for prototype)
        prev_y = y 
        
        # PACKET STRUCTURE
        # 1. Type (I or P)
        # 2. Color (Tiny Y + Cb + Cr)
        # 3. Mask (Bitmask of updates)
        # 4. Stream (Instructions)
        # 5. Codebook (Only if I-Frame)
        
        # Compress Components
        c_color = lzma.compress(y_tiny.astype(np.uint8).tobytes() + cr.tobytes() + cb.tobytes(), preset=1)
        c_mask = lzma.compress(mask.tobytes(), preset=1)
        c_stream = lzma.compress(stream.tobytes(), preset=1)
        
        frame_packet = {
            'type': 1 if is_iframe else 0,
            'color': c_color,
            'mask': c_mask,
            'stream': c_stream
        }
        
        if is_iframe:
            frame_packet['codebook'] = lzma.compress(encoder.codebook.astype(np.float16).tobytes(), preset=5)
            
        frames_data.append(frame_packet)
        if i % 10 == 0: print(f"Encoded frame {i}/{max_frames}...")

    cap.release()
    
    # WRITE FILE
    # Header: [Version] [W] [H] [FPS] [Count]
    # Frame Index: [Offset1] [Offset2] ...
    
    with open(out_file, 'wb') as f:
        f.write(VERSION)
        f.write(struct.pack('<HHfI', w, h, fps, len(frames_data)))
        
        # Write Frame Blobs
        for fd in frames_data:
            # Flags: Bit 0 = Is_IFrame
            flags = fd['type']
            
            # Write chunks
            blob = bytearray()
            blob.extend(struct.pack('<I', len(fd['color']))); blob.extend(fd['color'])
            blob.extend(struct.pack('<I', len(fd['mask']))); blob.extend(fd['mask'])
            blob.extend(struct.pack('<I', len(fd['stream']))); blob.extend(fd['stream'])
            
            if flags == 1:
                blob.extend(struct.pack('<I', len(fd['codebook']))); blob.extend(fd['codebook'])
                
            # Write frame header: [Flags] [TotalSize]
            f.write(struct.pack('<BI', flags, len(blob)))
            f.write(blob)

    total_size = os.path.getsize(out_file) / 1024
    print(f"Done. Video Size: {total_size:.2f} KB")
    return out_file

# --- DECODER & PLAYER ---
class FluxPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("ADI v30 Flux Player")
        self.root.geometry("1024x768")
        self.root.configure(bg="black")
        
        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.btn_frame = tk.Frame(root, bg="#222")
        self.btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Button(self.btn_frame, text="Load Video", command=self.load_video, bg="#444", fg="white").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(self.btn_frame, text="Play", command=self.play_video, bg="#00aa00", fg="white").pack(side=tk.LEFT, padx=10, pady=5)
        
        self.lbl_info = tk.Label(self.btn_frame, text="No Video", bg="#222", fg="gray")
        self.lbl_info.pack(side=tk.LEFT, padx=20)
        
        self.frames = []
        self.meta = {}
        self.playing = False
        
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv")])
        if path:
            self.lbl_info.config(text="Encoding... Please Wait")
            self.root.update()
            adv_path = encode_video(path)
            if adv_path:
                self.decode_video_memory(adv_path)
                
    def decode_video_memory(self, path):
        import lzma
        self.frames = []
        
        with open(path, 'rb') as f:
            ver = f.read(4)
            if ver != VERSION: return
            w, h, fps, count = struct.unpack('<HHfI', f.read(12))
            
            self.meta = {'w': w, 'h': h, 'fps': fps, 'count': count}
            
            current_codebook = None
            prev_y_image = np.zeros((h, w), dtype=np.uint8)
            
            # Precompute Grids
            xv, yv = np.meshgrid(np.linspace(-3.5, 3.5, 8), np.linspace(-3.5, 3.5, 8))
            xv, yv = xv.astype(np.float32).flatten(), yv.astype(np.float32).flatten()
            
            rows, cols = h//8, w//8
            
            print("Decoding to RAM...")
            for i in range(count):
                flags, size = struct.unpack('<BI', f.read(5))
                is_iframe = (flags == 1)
                
                # Read chunks
                sz_c = struct.unpack('<I', f.read(4))[0]; c_data = f.read(sz_c)
                sz_m = struct.unpack('<I', f.read(4))[0]; m_data = f.read(sz_m)
                sz_s = struct.unpack('<I', f.read(4))[0]; s_data = f.read(sz_s)
                
                # Update Codebook?
                if is_iframe:
                    sz_cb = struct.unpack('<I', f.read(4))[0]; cb_data = f.read(sz_cb)
                    current_codebook = np.frombuffer(lzma.decompress(cb_data), dtype=np.float16).astype(np.float32).reshape(-1, 64)
                
                # 1. Decode Color Base
                raw_c = lzma.decompress(c_data)
                pcs = (w//8)*(h//8)
                y_t = np.frombuffer(raw_c[:pcs], dtype=np.uint8).reshape(h//8, w//8)
                cr_t = np.frombuffer(raw_c[pcs:pcs*2], dtype=np.uint8).reshape(h//8, w//8)
                cb_t = np.frombuffer(raw_c[pcs*2:], dtype=np.uint8).reshape(h//8, w//8)
                
                y_pred = cv2.resize(y_t, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                cb = cv2.resize(cb_t, (w, h), interpolation=cv2.INTER_CUBIC)
                cr = cv2.resize(cr_t, (w, h), interpolation=cv2.INTER_CUBIC)
                
                # 2. Decode Stream
                mask = np.frombuffer(lzma.decompress(m_data), dtype=bool)
                stream = np.frombuffer(lzma.decompress(s_data), dtype=np.uint32)
                
                # 3. Reconstruct Detail
                # Start with previous frame (Temporal Persistence)
                # But we subtract the *previous* prediction to get the *previous* residual?
                # Actually, simpler: Reconstruct the *update* map and add to prediction
                
                # Initialize detail with zeros
                detail_flat = np.zeros((rows*cols, 64), dtype=np.float32)
                
                # Identify Update Types
                # mask is (N_Blocks,) bool. True = Updated.
                # stream corresponds to mask==True entries.
                
                if len(stream) > 0:
                    is_solid = (stream >> 31) & 1
                    
                    # Solids
                    mask_solid = (is_solid == 1)
                    if np.any(mask_solid):
                        s_vals = stream[mask_solid]
                        tok = (s_vals >> 14) & 0x3FF
                        sig = ((s_vals >> 8) & 0x3F).astype(np.float32)
                        mu  = (s_vals & 0xFF).astype(np.float32) - 128.0
                        
                        detail_flat[np.where(mask)[0][mask_solid]] = current_codebook[tok] * sig[:, None] + mu[:, None]

                    # Liquids
                    mask_liquid = (is_solid == 0)
                    if np.any(mask_liquid):
                        l_vals = stream[mask_liquid]
                        sx = (((l_vals >> 14) & 0x3F).astype(np.float32) / 8.0) - 4.0
                        sy = (((l_vals >> 8) & 0x3F).astype(np.float32) / 8.0) - 4.0
                        mu = (l_vals & 0xFF).astype(np.float32) - 128.0
                        
                        planes = (sx[:, None] * xv) + (sy[:, None] * yv) + mu[:, None]
                        detail_flat[np.where(mask)[0][mask_liquid]] = planes
                
                # 4. Composite
                detail_map = detail_flat.reshape(rows, cols, 8, 8).transpose(0, 2, 1, 3).reshape(h, w)
                
                # Handle temporal copy:
                # If mask is False, we should ideally keep the detail from prev frame.
                # However, since y_pred changes (it's from current frame low-res), 
                # we are essentially adding "0" detail to the new low-res base.
                # This works for "Liquid" logic (Base + Detail).
                
                y_raw = np.clip(y_pred + detail_map, 0, 255).astype(np.uint8)
                
                # Deblock
                grid_mask = np.zeros_like(y_raw)
                grid_mask[:, 8:w:8] = 255; grid_mask[8:h:8, :] = 255
                blurred = cv2.GaussianBlur(y_raw, (3,3), 0.5)
                y_raw[grid_mask==255] = blurred[grid_mask==255]
                
                # YCrCb -> RGB
                img = cv2.merge([y_raw, cr, cb])
                img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
                
                self.frames.append(ImageTk.PhotoImage(Image.fromarray(img)))
                if i%10==0: self.lbl_info.config(text=f"Decoded {i}/{count}")
                self.root.update()

            self.lbl_info.config(text=f"Ready: {count} Frames")

    def play_video(self):
        if not self.frames: return
        self.playing = True
        fps = self.meta.get('fps', 24)
        delay = int(1000 / fps)
        
        for frame in self.frames:
            if not self.playing: break
            self.canvas.create_image(0, 0, image=frame, anchor=tk.NW)
            self.root.update()
            time.sleep(delay / 1000.0)
        self.playing = False

if __name__ == "__main__":
    root = tk.Tk()
    app = FluxPlayer(root)
    root.mainloop()