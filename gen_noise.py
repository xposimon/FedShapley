import numpy as np
import os

def handle(x):
    return float(x.strip("[").strip("]").strip())
# noise = np.random.randn(28*28)

# with open("./noise.out", "w") as f:
#     f.write(str(list(noise)))

directory = os.path.dirname(__file__)
print(os.path.join(directory, "noise.out"))

with open("./noise.out", "r") as f:
    content = f.read()

noise = content.split(",")
noise = list(map(handle, noise))
print(len(noise))