from os import listdir
import random
from os.path import isfile, join
import sys
import os

mypath = sys.argv[1]
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)

for file in onlyfiles:
    old_path = mypath+"/"+file
    while True:
        new_file = str(random.randint(0, 100))
        new_path = mypath+"/"+new_file+".pdf"
        if not os.path.isfile(new_path):
            break
    os.rename(old_path,new_path)
