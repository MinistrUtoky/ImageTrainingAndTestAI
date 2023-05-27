#!/usr/bin/python
from PIL import Image
import os, sys

path = "Dataset/angry/"
dirs = os.listdir( path )

def removejpgs():
    for item in dirs:
        if os.path.isfile(path+item):
            f, e = os.path.splitext(path+item)
            if e==".jfif":
                os.remove(f + ".jfif")

removejpgs()