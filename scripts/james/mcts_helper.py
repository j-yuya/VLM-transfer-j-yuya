#!/usr/bin/env python3
"""
re-render_top_k.py – read top-k JSON and regenerate the PNGs
"""

import json, os, sys
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Literal
from PIL import Image                             # used only for type hints
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.functional as VF   # used for affine + InterpolationMode
from PIL import ImageDraw, ImageFont
# ──────────────────────────────────────────────────────────────────────────
#  Dataclass definitions (exactly the same as in your pipeline)
RGBA   = Tuple[int, int, int, int]
BBox   = Tuple[int, int, int, int]                 # (x0, y0, x1, y1)
Point  = Tuple[int, int]
Verts3 = Tuple[Point, Point, Point]

@dataclass(frozen=True, slots=True)
class ShapeMeta:
    kind:     Literal["rectangle", "ellipse", "triangle"]
    rgba:     RGBA
    bbox:     Optional[BBox]      = None
    vertices: Optional[Verts3]    = None
    angle:    Optional[float]     = None           # deg

@dataclass(frozen=True)
class AffineMeta:
    angle:      float               # deg
    translate:  Tuple[float, float] # px
    scale:      float               # scalar
    shear:      float = 0.          # deg
# ──────────────────────────────────────────────────────────────────────────

CANVAS_H, CANVAS_W = 512, 512           # change to match your model’s input
INSTR_FONT_SIZE    = 52                # “very big font”
INSTR_MARGIN       = 6
             # ← change this

# --------------------------------------------------------------------------
def loads_shapes(shape_dicts: list[dict]) -> list[ShapeMeta]:
    """Convert list of dicts into ShapeMeta objects."""
    shapes = []
    for d in shape_dicts:
        shapes.append(
            ShapeMeta(
                kind=d["kind"],
                rgba=tuple(d["rgba"]),
                bbox=tuple(d["bbox"]) if d["bbox"] is not None else None,
                vertices=(
                    tuple(map(tuple, d["vertices"])) if d["vertices"] else None
                ),
                angle=d.get("angle"),
            )
        )
    return shapes

def loads_affs(aff_dicts: list[dict]) -> list[AffineMeta]:
    """Convert list of dicts into AffineMeta objects."""
    return [
        AffineMeta(
            angle=a["angle"],
            translate=tuple(a["translate"]),
            scale=a["scale"],
            shear=a.get("shear", 0.0),
        )
        for a in aff_dicts
    ]

# --------------------------------------------------------------------------
def main(json_path: str, out_dir: str = "."):
    with open(json_path, "r", encoding="utf-8") as fp:
        records = json.load(fp)

    os.makedirs(out_dir, exist_ok=True)
    for rec in records:
        txt       = rec["txt"]
        shapes    = loads_shapes(rec["shapes"])
        affs      = loads_affs(rec["affs"])
        filename  = rec["filename"]

        pil = render_text_shapes_affines(txt, shapes, affs)

        full_path = os.path.join(out_dir, filename)
        pil.save(full_path)
        print(f"✓ saved {full_path}")

# --------------------------------------------------------------------------

def render_text_shapes_affines(txt: str,
                               shapes_meta: List[ShapeMeta],
                               affines:     List[AffineMeta]) -> Tuple[Image.Image, float]:
    """
    Deterministic renderer: draw text + shapes, then replay *all* stored affines.
    No RNG is consulted here!
    """
    blank = torch.ones(3, CANVAS_H, CANVAS_W)

    # text -------------------------------------------------------------
    tensor, _ = overlay_text_block(
        blank, txt,
        font_size=INSTR_FONT_SIZE,
        margin   =INSTR_MARGIN,
    )

    # shapes -----------------------------------------------------------
    tensor = replay_shapes(tensor, shapes_meta)

    # affines ----------------------------------------------------------
    pil = TF.to_pil_image(tensor)
    for a in affines:
        w, h  = pil.size
        dx, dy = a.translate
        pil = VF.affine(
            pil,
            angle=a.angle,
            translate=(dx * w, dy * h),   # convert back from relative to px
            scale=a.scale,
            shear=a.shear,
            interpolation=VF.InterpolationMode.BILINEAR,
            fill=255,
        )

    # loss -------------------------------------------------------------
    return pil

def overlay_text_block(
    img: torch.Tensor,
    text: str,
    *,
    font: Optional[ImageFont.ImageFont] = None,
    font_size: int = 9,
    line_spacing: float = 1.2,
    margin: int = 5,
) -> Tuple[torch.Tensor, dict]:
    """
    Same contract as before, but now honours explicit «\\n» line breaks.
    Returns an extra flag ``"truncated": bool`` in the metadata when
    the text could not fully fit inside the image height.
    """
    # --- Validate & convert image ------------------------------------------
    if not (isinstance(img, torch.Tensor) and img.ndim == 3):
        raise TypeError("`img` must be a CHW torch.Tensor")

    C, H, W = img.shape
    pil_base = TF.to_pil_image(img.clamp(0, 1)).convert("RGBA")

    # --- Font ---------------------------------------------------------------
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except OSError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_base)

    # --- Hard-line split ----------------------------------------------------
    para_lines: List[str] = text.split("\n")        # keep empty lines

    # --- Soft word-wrap each hard line -------------------------------------
    wrapped: List[str] = []
    for hard in para_lines:
        words = hard.split() if hard.strip() else [""]  # preserve blank line
        cur = ""
        for word in words:
            tentative = f"{cur} {word}".strip()
            w_px, _ = draw.textbbox((0, 0), tentative, font=font)[2:4]
            if w_px + 2 * margin <= W:
                cur = tentative
            else:
                if cur:
                    wrapped.append(cur)
                cur = word
        if cur or hard == "":
            wrapped.append(cur)

    # --- Render -------------------------------------------------------------
    y = margin
    positions: List[Tuple[int, int]] = []
    truncated = False

    for line in wrapped:
        if y > H - font_size:
            truncated = True
            break
        draw.text((margin, y), line, fill=(0, 0, 0, 255), font=font)
        positions.append((margin, y))
        y += int(font_size * line_spacing)

    tensor_out = TF.to_tensor(pil_base.convert("RGB"))

    return tensor_out, {
        "font_size": font_size,
        "line_spacing": line_spacing,
        "margin": margin,
        "lines": wrapped,
        "positions": positions,
        "truncated": truncated,  # new field
    }

def replay_shapes(
    base: torch.Tensor | None,
    shapes_meta: list[ShapeMeta],
    size: tuple[int, int] | None = None,
) -> torch.Tensor:

    if base is None and size is None:
        raise ValueError("Need `size` when no base is supplied")

    if base is None:
        H, W = size
        base = torch.ones(3, H, W)
    else:
        _, H, W = base.shape

    canvas = TF.to_pil_image(base.clamp(0, 1)).convert("RGBA")

    for s in shapes_meta:
        kind  = s.kind
        col   = tuple(s.rgba)

        # ---------- TRIANGLE --------------------------------------------
        if kind == "triangle":
            verts = (
                s.vertices
                if s.vertices is not None
                else [(s.bbox[0], s.bbox[3]),
                      ((s.bbox[0]+s.bbox[2])//2, s.bbox[1]),
                      (s.bbox[2], s.bbox[3])]
            )
            ImageDraw.Draw(canvas, "RGBA").polygon(verts, fill=col)
            continue

        # ---------- RECT / ELLIPSE (maybe rotated) ----------------------
        bbox  = s.bbox
        angle = s.angle

        if angle is None:
            d = ImageDraw.Draw(canvas, "RGBA")
            if kind == "rectangle":
                d.rectangle(bbox, fill=col)
            elif kind == "ellipse":
                d.ellipse(bbox, fill=col)
        else:
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            d_ov    = ImageDraw.Draw(overlay, "RGBA")
            if kind == "rectangle":
                d_ov.rectangle(bbox, fill=col)
            elif kind == "ellipse":
                d_ov.ellipse(bbox, fill=col)

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            rotated = overlay.rotate(angle, center=(cx, cy), expand=False)
            canvas  = Image.alpha_composite(canvas, rotated)

    return TF.to_tensor(canvas.convert("RGB"))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("usage: re-render_top_k.py top_k_results.json [output_dir]")
        sys.exit(1)

    json_file  = sys.argv[1]
    out_folder = sys.argv[2] if len(sys.argv) == 3 else "."
    main(json_file, out_folder)