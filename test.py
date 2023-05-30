import pickle
import json

with open('resdata','rb') as f:
    res = pickle.loads(f.read())
    
    with open('resdata.json','w') as fs:
        fs.write(str(res))