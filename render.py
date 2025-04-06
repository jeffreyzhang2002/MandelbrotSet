#!/usr/bin/env python3

from PIL import Image
import numpy as np
import sys

if len(sys.argv) != 3:
    print(f"{sys.argv[0]} <bitmap file> <outputfile> <black_threshold>")

input_file = sys.argv[1]
output_file = sys.argv[2]
threshold = int(sys.argv[3])

print(f"Input: {input_file}")
print(f"Output: {output_file}")

file = open(input_file, "r")

metadata = file.readline().split(" ")
width=int(metadata[1])
height=int(metadata[2])

print(f"Dimensions {width}, {height}")

length = width*height
pixels=np.empty((height, width, 3), dtype=np.uint8)

BLACK = np.array([0, 0, 0])
WHITE = np.array([255, 255, 255])



for i in range(width):
    for j in range(height):
        value = int(file.readline())            
        if value >= threshold:
            pixels[i][j] = BLACK
        else:
            if value < 10:
                pixels[i][j] = WHITE 
            else:
                pixels[i][j] = np.array([value * 30, value * 30, value])

image = Image.fromarray(pixels, 'RGB')
image.save(output_file)
