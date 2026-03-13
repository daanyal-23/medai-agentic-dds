"""Draw bounding box overlays on PIL images."""
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "high":   "#e74c3c",
    "medium": "#f39c12",
    "low":    "#2ecc71",
    "default":"#3498db"
}


def draw_overlays(image: Image.Image, overlays: list[dict]) -> Image.Image:
    if not overlays:
        return image

    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size

    for ovl in overlays:
        coords = ovl.get("coords", [])
        if len(coords) != 4:
            continue

        xf, yf, wf, hf = coords
        # Support both fractional (0-1) and absolute pixel coords
        if all(0 <= v <= 1 for v in coords):
            x1 = int(xf * w);        y1 = int(yf * h)
            x2 = int((xf + wf) * w); y2 = int((yf + hf) * h)
        else:
            x1, y1, x2, y2 = int(xf), int(yf), int(xf+wf), int(yf+hf)

        finding = ovl.get("finding","")
        label   = ovl.get("label", finding)

        # Color by confidence if available
        color = COLORS["default"]

        # Draw semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(231, 76, 60, 40), outline=color, width=3)

        # Draw label background
        label_h = 20
        draw.rectangle([x1, y1 - label_h, x1 + len(label) * 8 + 8, y1], fill=color)
        try:
            draw.text((x1 + 4, y1 - label_h + 2), label, fill="white")
        except Exception:
            pass

    return image
