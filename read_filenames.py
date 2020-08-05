import numpy as np

numbers = []
with open('results/filenames.txt', encoding="utf-8") as f:
    for line in f:
        __,__, filE = line.split("_") #  line.split("\t") if numbers are seperated by tab
        number,__ = filE.split(".")
        numbers.append(int(number))

numbers = np.array(numbers)