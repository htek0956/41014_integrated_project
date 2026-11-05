import cv2 as cv
import numpy as np
from pathlib import Path

NUM_MARKERS = 30
PIXELS = 200


output_folder = Path(__file__).resolve().parent.parent / "assets" / "markers"
output_folder.mkdir(parents=True, exist_ok=True)


dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

for i in range(NUM_MARKERS):


    marker_image = cv.aruco.generateImageMarker(dictionary, i, PIXELS)

    filename = output_folder / f"{i}.png"

    if not cv.imwrite(str(filename), marker_image):
        raise OSError(f"Could not save {filename}")
