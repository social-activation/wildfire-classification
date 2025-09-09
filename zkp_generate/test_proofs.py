import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk, ImageOps
import numpy as np
import json
import os
import time

# Optional deps for proving
try:
    import ezkl
    from pytictoc import TicToc
    EZKL_AVAILABLE = True
except Exception:
    EZKL_AVAILABLE = False

# ----- App config -----
DISPLAY_MAX = 500                  # Max px for original preview
RESIZED_DISPLAY_SCALE = 4          # 64x64 shown as 256x256 for clarity
JSON_NORMALIZE = True              # Output floats in [0,1]; set False for 0..255
DEFAULT_INPUT_JSON = "input_image_rgb_64x64.json"
PROOFS_DIRNAME = "proofs"
SRS_PATH = os.path.expanduser("~/.ezkl/srs/kzg17.srs")   # auto-created if missing
SRS_LOG2_DEGREE = 18                                      # matches your snippet

def log_safe(widget, msg):
    """Append text to a ScrolledText widget safely."""
    if not widget:
        print(msg)
        return
    widget.insert(tk.END, msg + "\n")
    widget.see(tk.END)
    widget.update_idletasks()

def find_first(paths):
    """Return the first existing path from a list, else None."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def search_for(filename, start_dir):
    """Walk the tree to find the first occurrence of filename."""
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def autodiscover_artifacts(base_dir, logger=None):
    """
    Discover EZKL artifacts near app.py (base_dir):
      - compiled model: network.ezkl or network.compiled
      - settings.json, key.pk, key.vk
    Searches base_dir and subfolders.
    """
    compiled = find_first([
        os.path.join(base_dir, "network.ezkl"),
        os.path.join(base_dir, "network.compiled"),
    ]) or search_for("network.ezkl", base_dir) or search_for("network.compiled", base_dir)

    settings = find_first([
        os.path.join(base_dir, "settings.json"),
    ]) or search_for("settings.json", base_dir)

    pk = find_first([
        os.path.join(base_dir, "key.pk"),
    ]) or search_for("key.pk", base_dir)

    vk = find_first([
        os.path.join(base_dir, "key.vk"),
    ]) or search_for("key.vk", base_dir)

    if logger:
        log_safe(logger, f"[auto] compiled_model: {compiled}")
        log_safe(logger, f"[auto] settings     : {settings}")
        log_safe(logger, f"[auto] pk          : {pk}")
        log_safe(logger, f"[auto] vk          : {vk}")

    return compiled, settings, pk, vk

def ensure_srs(srs_path, k, logger=None):
    """Create the SRS file if missing (ezkl.gen_srs)."""
    if not EZKL_AVAILABLE:
        return False
    srs_dir = os.path.dirname(srs_path)
    os.makedirs(srs_dir, exist_ok=True)
    if not os.path.exists(srs_path):
        if logger:
            log_safe(logger, f"[ezkl] SRS not found; generating: {srs_path} (k={k})")
        ezkl.gen_srs(srs_path, k)
    else:
        if logger:
            log_safe(logger, f"[ezkl] SRS already present: {srs_path}")
    return True

def infer_and_build_proof(compiled_model_path, settings_path, pk_path, vk_path,
                          input_path, output_path, logger=None):
    """
    Runs witness generation and proving synchronously (single proof).
    settings_path and vk_path are passed for interface parity.
    """
    if not EZKL_AVAILABLE:
        raise RuntimeError("ezkl is not installed/available in this environment.")

    witness_path = "witness.json"
    t = TicToc()

    # 1) Generate witness
    if logger: log_safe(logger, f"[ezkl] gen_witness → {witness_path}")
    t.tic()
    _ = ezkl.gen_witness(
        input_path,
        compiled_model_path,
        witness_path,
        vk_path,
        settings_path
    )
    elapsed_gen_witness = t.tocvalue()
    if logger: log_safe(logger, f"[timing] Gen Witness: {elapsed_gen_witness:.3f}s")

    # 2) Prove
    if logger: log_safe(logger, f"[ezkl] prove → {output_path}")
    t.tic()
    _ = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        output_path,
        "single",
    )
    elapsed_prove = t.tocvalue()
    if logger:
        log_safe(logger, f"[timing] Prove: {elapsed_prove:.3f}s")
        log_safe(logger, "[ezkl] Done.")

    assert os.path.isfile(output_path), f"Proof was not written to {output_path}"
    try:
        os.remove(witness_path)
    except Exception:
        pass

class ImageToTensorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image → RGB 3×64×64 + JSON + EZKL Proof")
        self.minsize(1100, 760)

        # Keep references so Tk doesn't GC images
        self._orig_photo = None
        self._resized_photo = None

        # Data holders
        self.tensor_chw = None           # (3, 64, 64) RGB uint8
        self.json_str = None             # JSON string of RGB flattened (1×12288)
        self.input_image_path = None     # selected image path
        self.input_json_path = None      # where we saved the JSON (next to app.py)

        # ---- Top bar ----
        top = tk.Frame(self, padx=10, pady=10)
        top.pack(fill=tk.X)

        tk.Button(top, text="Select Image…", command=self.select_image).pack(side=tk.LEFT)

        self.info_var = tk.StringVar(value="No image loaded")
        tk.Label(top, textvariable=self.info_var, anchor="w").pack(side=tk.LEFT, padx=12)

        # ---- Image previews ----
        body = tk.Frame(self, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="Original").pack(anchor="w")
        self.orig_label = tk.Label(left, bd=1, relief=tk.SOLID, width=60, height=30)
        self.orig_label.pack(fill=tk.BOTH, expand=True, padx=(0, 8), pady=4)

        right = tk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right, text="Converted (RGB 64×64; display scaled)").pack(anchor="w")
        self.resized_label = tk.Label(right, bd=1, relief=tk.SOLID, width=60, height=30)
        self.resized_label.pack(fill=tk.BOTH, expand=True, padx=(8, 8), pady=4)

        # ---- JSON tools ----
        json_box_frame = tk.Frame(self, padx=10, pady=0)
        json_box_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            json_box_frame,
            text="JSON preview (format: {\"input_data\": [[...12288 floats (RGB, CHW, flattened)...]]})"
        ).pack(anchor="w")

        self.json_text = ScrolledText(json_box_frame, height=10, wrap="none")
        self.json_text.pack(fill=tk.BOTH, expand=True)

        # ---- Prover controls ----
        controls = tk.Frame(self, padx=10, pady=10)
        controls.pack(fill=tk.X)

        self.btn_save_json = tk.Button(controls, text="Save JSON…", command=self.save_json, state=tk.DISABLED)
        self.btn_save_json.pack(side=tk.LEFT)

        self.btn_gen_proof = tk.Button(controls, text="Generate Proof", command=self.generate_proof, state=tk.NORMAL)
        self.btn_gen_proof.pack(side=tk.LEFT, padx=8)

        # ---- Log output ----
        log_frame = tk.LabelFrame(self, text="Logs", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = ScrolledText(log_frame, height=12, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        if not EZKL_AVAILABLE:
            log_safe(self.log_text, "[warn] ezkl / pytictoc not available. You can still make JSON, but proving is disabled.")
            self.btn_gen_proof.config(state=tk.NORMAL)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return
        try:
            # Load & orient
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)   # respect camera EXIF orientation
            orig_w, orig_h = img.size

            # --- Make RGB 64×64 (no grayscale) ---
            img_rgb = img.convert("RGB")
            resized_64_rgb = img_rgb.resize((64, 64), Image.LANCZOS)

            # (3,64,64) tensor, uint8 channel-first
            arr_rgb_u8 = np.array(resized_64_rgb, dtype=np.uint8)    # (64,64,3)
            self.tensor_chw = np.transpose(arr_rgb_u8, (2, 0, 1))    # -> (3,64,64)

            # --- JSON payload: flatten RGB (CHW) to 1×12288 ---
            if JSON_NORMALIZE:
                arr = self.tensor_chw.astype(np.float32) / 255.0     # [0,1] floats
            else:
                arr = self.tensor_chw.astype(np.float32)             # 0..255 floats

            flat = arr.reshape(-1)                                   # length 12288 (3*64*64)
            payload = {"input_data": [flat.tolist()]}                # 1×12288

            self.json_str = json.dumps(payload, indent=2, ensure_ascii=False)
            self.input_image_path = path

            # --- Update previews ---
            # Original (fit)
            orig_display = img.copy()
            orig_display.thumbnail((DISPLAY_MAX, DISPLAY_MAX), Image.LANCZOS)
            self._orig_photo = ImageTk.PhotoImage(orig_display)
            self.orig_label.configure(image=self._orig_photo)

            # RGB 64×64 (scaled up for visibility)
            disp_size = (64 * RESIZED_DISPLAY_SCALE, 64 * RESIZED_DISPLAY_SCALE)
            rgb_display = resized_64_rgb.resize(disp_size, Image.NEAREST)
            self._resized_photo = ImageTk.PhotoImage(rgb_display)
            self.resized_label.configure(image=self._resized_photo)

            # Info + JSON box
            self.info_var.set(
                f"Original: {orig_w}×{orig_h} | Tensor (RGB CHW): {self.tensor_chw.shape} uint8 "
                f"| JSON length: {flat.size} floats ({'normalized' if JSON_NORMALIZE else '0..255'})"
            )
            self.json_text.delete("1.0", tk.END)
            self.json_text.insert(tk.END, self.json_str)

            self.btn_save_json.config(state=tk.NORMAL)
            self.btn_gen_proof.config(state=tk.NORMAL if EZKL_AVAILABLE else tk.DISABLED)

            log_safe(self.log_text, f"[app] Prepared JSON for: {os.path.basename(path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not open/process image:\n{e}")

    def save_json(self):
        if not self.json_str:
            messagebox.showwarning("No JSON", "Load an image first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save JSON",
            defaultextension=".json",
            initialfile=DEFAULT_INPUT_JSON,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(self.json_str)
            messagebox.showinfo("Saved", f"Saved JSON to:\n{os.path.abspath(out_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save JSON:\n{e}")

    def _write_input_json_next_to_app(self):
        """Persist the current JSON next to app.py (working directory)."""
        if not self.json_str:
            raise RuntimeError("No JSON prepared.")
        self.input_json_path = os.path.join(os.getcwd(), DEFAULT_INPUT_JSON)
        with open(self.input_json_path, "w", encoding="utf-8") as f:
            f.write(self.json_str)
        log_safe(self.log_text, f"[app] Wrote input JSON → {self.input_json_path}")

    def generate_proof(self):
        """Generate a proof using EZKL for the selected image’s JSON."""
        if not EZKL_AVAILABLE:
            messagebox.showwarning("EZKL missing", "ezkl not available. Install ezkl to enable proving.")
            return
        if not self.json_str:
            messagebox.showwarning("No JSON", "Load an image to create JSON first.")
            return

        # 1) Save the input JSON beside app.py automatically
        try:
            self._write_input_json_next_to_app()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write input JSON:\n{e}")
            return

        # 2) Autodiscover artifacts next to app.py (or in subfolders)
        base_dir = os.getcwd()
        compiled_model_path, settings_path, pk_path, vk_path = autodiscover_artifacts(base_dir, self.log_text)

        missing = [("compiled model (network.ezkl/.compiled)", compiled_model_path),
                   ("settings.json", settings_path),
                   ("key.pk", pk_path),
                   ("key.vk", vk_path)]
        problems = [name for name, p in missing if not p]
        if problems:
            messagebox.showerror(
                "Artifacts not found",
                "Could not find the following near app.py:\n- " + "\n- ".join(problems) +
                "\n\nPlace these files next to app.py (or in a subfolder) and try again."
            )
            return

        # 3) Ensure SRS
        try:
            ensure_srs(SRS_PATH, SRS_LOG2_DEGREE, self.log_text)
        except Exception as e:
            log_safe(self.log_text, f"[warn] SRS generation failed or skipped: {e}")

        # 4) Proof path (proofs/<image_base>_proof.pf)
        proofs_dir = os.path.join(base_dir, PROOFS_DIRNAME)
        os.makedirs(proofs_dir, exist_ok=True)
        image_base = os.path.splitext(os.path.basename(self.input_image_path or "input"))[0]
        proof_path = os.path.join(proofs_dir, f"{image_base}_proof.pf")

        # 5) Run proving
        start = time.time()
        try:
            infer_and_build_proof(
                compiled_model_path=compiled_model_path,
                settings_path=settings_path,
                pk_path=pk_path,
                vk_path=vk_path,
                input_path=self.input_json_path,
                output_path=proof_path,
                logger=self.log_text,
            )
        except Exception as e:
            log_safe(self.log_text, f"[ERROR] Proof generation failed: {e}")
            messagebox.showerror("Proving failed", f"Failed to create proof:\n{e}")
            return

        elapsed = time.time() - start
        log_safe(self.log_text, f"[done] Proof created at: {proof_path}")
        log_safe(self.log_text, f"[timing] Total: {elapsed:.3f}s")
        messagebox.showinfo("Proof created", f"Proof written to:\n{proof_path}")

if __name__ == "__main__":
    app = ImageToTensorApp()
    app.mainloop()
