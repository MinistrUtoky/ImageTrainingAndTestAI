#!/usr/bin/python
from PIL import Image
import os, sys
import string, random

path = "Dataset/angry/"
dirs = os.listdir( path )

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def resize():
    i = 0
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(path + id_generator() + str(i) + '.png', 'PNG', quality=90)
            os.remove(f + ".png")
            i+=1

resize()