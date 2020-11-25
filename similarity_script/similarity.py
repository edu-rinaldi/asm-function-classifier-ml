from scipy.spatial.distance import hamming, cosine
import numpy as np
from sys import argv

if __name__ == "__main__":
    if len(argv) < 3:
        print("ERROR: v1 v2 args needed")
        exit(-1)
    
    valByClass = {'encryption' : 0, 'sort' : 1, 'math' : 2, 'string': 3}

    # read from file
    a = np.loadtxt(argv[1], dtype=int, converters={0: lambda x: valByClass[x]}, encoding='utf-8')
    b = np.loadtxt(argv[2], dtype=int, converters={0: lambda x: valByClass[x]}, encoding='utf-8')

    # calc...
    ham_res = hamming(a, b)
    cosine_res = cosine(a, b)

    # 0 if equal
    print(f"Hamming distance: {ham_res}")
    # 1 if equal
    print(f"Cosine similarity: {1 - cosine_res}")

