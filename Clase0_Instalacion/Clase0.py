import os
import torch
import fastai

print("Version de python: ", end='')
os.system("python --version")

print("Version de pytorch: ", end='')
print(torch.__version__)

print("Version de fastai: ", end='')
print(fastai.__version__)
