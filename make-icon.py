"""Готовит иконки приложения из исходной картинки assets/vibe2.jpg.

Что делает:
  - находит сам значок на фоне и обрезает поля;
  - делает квадрат 1024x1024;
  - скругляет углы и добавляет прозрачность (JPEG её не умеет);
  - сохраняет assets/icon.png (256) и assets/icon.ico (16…256).
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
SOURCE = os.path.join(ASSETS, "vibe2.jpg")
CORNER_RADIUS_RATIO = 0.175  # как у исходного значка


def find_content_box(img: Image.Image) -> tuple[int, int, int, int]:
    """Границы значка на тёмном фоне (без мягкой тени)."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    bg = np.array([arr[2, 2], arr[2, -3], arr[-3, 2], arr[-3, -3]]).mean(axis=0)
    distance = np.abs(arr - bg).sum(axis=2)
    mask = distance > 120  # тень слабее, значок заметно светлее фона
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, img.width, img.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def square_crop(img: Image.Image, box) -> Image.Image:
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    half = side // 2
    left, top = cx - half, cy - half
    right, bottom = left + side, top + side
    # если вышли за границы — сдвигаем окно внутрь
    left = max(0, min(left, img.width - side))
    top = max(0, min(top, img.height - side))
    return img.crop((left, top, left + side, top + side))


def rounded(img: Image.Image, radius_ratio: float = CORNER_RADIUS_RATIO) -> Image.Image:
    img = img.convert("RGBA")
    size = img.width
    radius = int(size * radius_ratio)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    out = img.copy()
    out.putalpha(mask)
    return out


def main() -> int:
    if not os.path.exists(SOURCE):
        raise SystemExit(f"нет исходника: {SOURCE}")
    src = Image.open(SOURCE)
    box = find_content_box(src)
    print(f"исходник {src.size}, значок в границах {box}")

    icon = rounded(square_crop(src, box).resize((1024, 1024), Image.LANCZOS))

    png_path = os.path.join(ASSETS, "icon.png")
    icon.resize((256, 256), Image.LANCZOS).save(png_path, format="PNG")
    print(f"сохранено: {png_path}")

    ico_path = os.path.join(ASSETS, "icon.ico")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    icon.resize((256, 256), Image.LANCZOS).save(
        ico_path, format="ICO", sizes=[(s, s) for s in sizes]
    )
    print(f"сохранено: {ico_path} ({', '.join(str(s) for s in sizes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
