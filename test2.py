import pickle

with open('resdata','rb') as f:
    print(pickle.loads(f.read()))