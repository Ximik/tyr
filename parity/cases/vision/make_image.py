"""Write the images of the vision cases: the same colored shapes on a light
background, drawn at each size. Needs Pillow:

    python parity/cases/vision/make_image.py"""

from pathlib import Path

from PIL import Image, ImageDraw

SIZES = {"shapes": (448, 336), "wide": (640, 224), "tall": (672, 896)}

for name, (width, height) in SIZES.items():
    def at(*points):
        return [(x * width / 448, y * height / 336) for x, y in points]

    image = Image.new("RGB", (width, height), (235, 232, 222))
    draw = ImageDraw.Draw(image)
    draw.rectangle(at((40, 50), (180, 170)), fill=(200, 40, 40))
    draw.ellipse(at((230, 40), (400, 210)), fill=(40, 90, 200))
    draw.polygon(at((90, 300), (200, 200), (310, 300)), fill=(40, 160, 70))
    draw.line(at((20, 320), (430, 250)), fill=(30, 30, 30), width=6)
    image.save(Path(__file__).resolve().parent / f"{name}.png")
