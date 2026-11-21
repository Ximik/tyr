"""Write `shapes.png`, the image of the vision cases: a few colored shapes on a
light background. Needs Pillow:

    python parity/cases/vision/make_image.py"""

from pathlib import Path

from PIL import Image, ImageDraw

image = Image.new("RGB", (448, 336), (235, 232, 222))
draw = ImageDraw.Draw(image)
draw.rectangle((40, 50, 180, 170), fill=(200, 40, 40))
draw.ellipse((230, 40, 400, 210), fill=(40, 90, 200))
draw.polygon([(90, 300), (200, 200), (310, 300)], fill=(40, 160, 70))
draw.line((20, 320, 430, 250), fill=(30, 30, 30), width=6)
image.save(Path(__file__).resolve().parent / "shapes.png")
