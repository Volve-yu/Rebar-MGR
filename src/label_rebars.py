"""
Manual rebar labeling tool.

Click on each vertical rebar center to place a red line.
Right-click or press Z to undo the last line.
Press S or Enter to save and move to the next image.
Press Q or Escape to quit.

Usage:
    python src/label_rebars.py --data_root data --subset site-1 --split trainval
    python src/label_rebars.py --data_root data --subset site-1 --split trainval --resume

Output:
    data/{subset}/rebar_labels/{image_id}.txt
    One line per rebar: x_pixel (integer, column of rebar center)
    Lines are sorted left to right.

The labels can be loaded by gt_report.py with --use_manual_labels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("tkinter is required (usually bundled with Python).")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Label I/O
# ---------------------------------------------------------------------------

def load_label(label_path: Path):
    """Return sorted list of x-pixel ints, or [] if file missing."""
    if not label_path.exists():
        return []
    xs = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    xs.append(int(line))
                except ValueError:
                    pass
    return sorted(xs)


def save_label(label_path: Path, xs: list):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for x in sorted(xs):
            f.write(f"{x}\n")


# ---------------------------------------------------------------------------
# Main labeling app
# ---------------------------------------------------------------------------

DISPLAY_W = 900   # canvas display width (image is scaled to fit)
DISPLAY_H = 700


class LabelApp:
    def __init__(self, root: tk.Tk, image_paths: list, label_dir: Path,
                 depth_labels: dict = None, cfg=None):
        self.root = root
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.depth_labels = depth_labels or {}
        self.cfg = cfg

        self.idx = 0          # current image index
        self.xs = []          # x-pixel positions of placed lines (display coords)
        self.xs_orig = []     # same in original image coords
        self.scale = 1.0
        self.img_w = 1
        self.img_h = 1

        self._build_ui()
        self._load_image()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root.title("Rebar Labeler")
        self.root.resizable(False, False)

        # Top info bar
        self.info_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.info_var, anchor="w",
                 font=("Courier", 10)).pack(fill="x", padx=4, pady=2)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=DISPLAY_W, height=DISPLAY_H,
                                bg="black", cursor="crosshair")
        self.canvas.pack()

        # Bottom status bar
        self.status_var = tk.StringVar(value="Click to place lines | Z=undo | S=save+next | Q=quit")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 font=("Courier", 9), fg="gray").pack(fill="x", padx=4)

        # Spacing info bar
        self.spacing_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.spacing_var, anchor="w",
                 font=("Courier", 9), fg="blue").pack(fill="x", padx=4, pady=2)

        # Bindings
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_undo)
        self.root.bind("<z>", self._on_undo)
        self.root.bind("<Z>", self._on_undo)
        self.root.bind("<s>", self._on_save_next)
        self.root.bind("<S>", self._on_save_next)
        self.root.bind("<Return>", self._on_save_next)
        self.root.bind("<q>", self._on_quit)
        self.root.bind("<Q>", self._on_quit)
        self.root.bind("<Escape>", self._on_quit)
        self.root.bind("<Left>", self._on_prev)
        self.root.bind("<Right>", self._on_save_next)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(self):
        if self.idx >= len(self.image_paths):
            messagebox.showinfo("Done", "All images labeled!")
            self.root.quit()
            return

        p = self.image_paths[self.idx]
        img = Image.open(p).convert("RGB")
        self.img_w, self.img_h = img.size

        # Scale to fit display
        scale_w = DISPLAY_W / self.img_w
        scale_h = DISPLAY_H / self.img_h
        self.scale = min(scale_w, scale_h, 1.0)   # never upscale
        dw = int(self.img_w * self.scale)
        dh = int(self.img_h * self.scale)

        self._display_img = img.resize((dw, dh), Image.LANCZOS)
        self._tk_img = None   # will be set in _redraw

        # Load existing label if any
        label_path = self.label_dir / f"{p.stem}.txt"
        saved_xs = load_label(label_path)
        self.xs_orig = saved_xs[:]
        self.xs = [int(x * self.scale) for x in saved_xs]

        self._update_info()
        self._redraw()

    def _update_info(self):
        p = self.image_paths[self.idx]
        sid = p.stem
        dist_str = ""
        if sid in self.depth_labels:
            d = self.depth_labels[sid]
            mpp = None
            if self.cfg:
                import math
                mpp = (2.0 * d * math.tan(math.radians(self.cfg.hfov_deg) / 2.0)
                       / self.cfg.image_width_px * 1000.0)
            dist_str = f"  dist={d}m" + (f"  {mpp:.2f}mm/px" if mpp else "")
        self.info_var.set(
            f"[{self.idx+1}/{len(self.image_paths)}]  {p.parent.parent.name}/{p.stem}{dist_str}"
            f"  |  lines placed: {len(self.xs)}"
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        from PIL import ImageTk, ImageDraw as PilDraw

        # Composite: base image + red lines + spacing labels
        composite = self._display_img.copy()
        draw = PilDraw.Draw(composite)
        dw, dh = composite.size

        sorted_xs = sorted(self.xs)
        for x in sorted_xs:
            draw.line([(x, 0), (x, dh)], fill=(255, 0, 0), width=2)

        # Spacing labels between adjacent lines
        if len(sorted_xs) >= 2 and self.cfg and self.img_w:
            import math
            sid = self.image_paths[self.idx].stem
            if sid in self.depth_labels:
                d = self.depth_labels[sid]
                mpp = (2.0 * d * math.tan(math.radians(self.cfg.hfov_deg) / 2.0)
                       / self.cfg.image_width_px * 1000.0)
                for i in range(len(sorted_xs) - 1):
                    x1, x2 = sorted_xs[i], sorted_xs[i + 1]
                    gap_px = (x2 - x1) / self.scale   # back to original px
                    gap_mm = gap_px * mpp
                    mid = (x1 + x2) // 2
                    draw.text((mid - 18, dh // 2 - 10), f"{gap_mm:.0f}mm",
                              fill=(0, 220, 0))

        self._tk_img = ImageTk.PhotoImage(composite)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        # Update spacing summary
        self._update_spacing_summary()

    def _update_spacing_summary(self):
        if len(self.xs_orig) < 2 or not self.cfg:
            self.spacing_var.set("")
            return
        import math
        sid = self.image_paths[self.idx].stem
        if sid not in self.depth_labels:
            self.spacing_var.set("")
            return
        d = self.depth_labels[sid]
        mpp = (2.0 * d * math.tan(math.radians(self.cfg.hfov_deg) / 2.0)
               / self.cfg.image_width_px * 1000.0)
        sxs = sorted(self.xs_orig)
        gaps = [(sxs[i+1] - sxs[i]) * mpp for i in range(len(sxs)-1)]
        mean_g = sum(gaps) / len(gaps)
        std_g = (sum((g - mean_g)**2 for g in gaps) / len(gaps)) ** 0.5
        gaps_str = "  ".join(f"{g:.0f}" for g in gaps)
        self.spacing_var.set(
            f"spacings(mm): [{gaps_str}]   mean={mean_g:.1f}  std={std_g:.1f}"
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_click(self, event):
        x_disp = event.x
        x_orig = int(x_disp / self.scale)
        x_orig = max(0, min(x_orig, self.img_w - 1))
        x_disp_snapped = int(x_orig * self.scale)

        self.xs.append(x_disp_snapped)
        self.xs_orig.append(x_orig)
        self._update_info()
        self._redraw()

    def _on_undo(self, event=None):
        if self.xs:
            self.xs.pop()
            self.xs_orig.pop()
            self._update_info()
            self._redraw()

    def _on_save_next(self, event=None):
        p = self.image_paths[self.idx]
        label_path = self.label_dir / f"{p.stem}.txt"
        save_label(label_path, self.xs_orig)
        n = len(self.xs_orig)
        self.status_var.set(f"Saved {n} lines for {p.stem}. Loading next...")
        self.root.update()
        self.idx += 1
        self._load_image()

    def _on_prev(self, event=None):
        if self.idx > 0:
            # Save current first
            p = self.image_paths[self.idx]
            label_path = self.label_dir / f"{p.stem}.txt"
            save_label(label_path, self.xs_orig)
            self.idx -= 1
            self._load_image()

    def _on_quit(self, event=None):
        if messagebox.askyesno("Quit", "Save current image and quit?"):
            p = self.image_paths[self.idx]
            label_path = self.label_dir / f"{p.stem}.txt"
            save_label(label_path, self.xs_orig)
        self.root.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Manual rebar line labeling tool")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--subset", default="site-1", choices=["site-1", "site-2"])
    ap.add_argument("--split", default="trainval",
                    choices=["train", "val", "trainval"])
    ap.add_argument("--resume", action="store_true",
                    help="Skip images that already have a label file")
    ap.add_argument("--hfov", type=float, default=87.0)
    ap.add_argument("--img_w", type=int, default=640)
    ap.add_argument("--img_h", type=int, default=480)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    sub_path = data_root / args.subset
    img_dir = sub_path / "images"
    label_dir = sub_path / "rebar_labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    # Load split list
    split_file = sub_path / "gt_sets" / f"{args.split}.txt"
    if split_file.exists():
        ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    else:
        ids = [p.stem for p in sorted(img_dir.glob("*.png"))]

    image_paths = []
    for sid in ids:
        p = img_dir / f"{sid}.png"
        if not p.exists():
            continue
        if args.resume and (label_dir / f"{sid}.txt").exists():
            continue
        image_paths.append(p)

    if not image_paths:
        print("No images to label (all done or none found).")
        return

    print(f"Images to label: {len(image_paths)}")

    # Load depth labels for mm display
    depth_labels = {}
    depth_file = sub_path / "depth_labels.txt"
    if depth_file.exists():
        from measure import load_depth_labels, RulerConfig
        depth_labels = load_depth_labels(str(depth_file))
        cfg = RulerConfig(hfov_deg=args.hfov, image_width_px=args.img_w,
                          image_height_px=args.img_h)
    else:
        cfg = None

    root = tk.Tk()
    app = LabelApp(root, image_paths, label_dir, depth_labels, cfg)
    root.mainloop()

    labeled = len(list(label_dir.glob("*.txt")))
    print(f"Done. {labeled} label files in {label_dir}")


if __name__ == "__main__":
    main()
