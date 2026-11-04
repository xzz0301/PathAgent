"""Encode only the frozen pilot pool with original PLIP weights."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.project/'external/PLIP'));from plip import PLIP
    model=PLIP(str(a.project/'checkpoints/plip'))
    for folder in sorted((a.input/'patches').iterdir()):
        dest=a.input/'features'/folder.name;dest.mkdir(parents=True,exist_ok=True)
        paths=sorted(folder.glob('*.png'))
        for offset in range(0,len(paths),16):
            batch=paths[offset:offset+16];images=[]
            for f in batch:
                with Image.open(f) as im:images.append(im.convert('RGB'))
            embeddings=model.encode_images(images,batch_size=len(images));embeddings/=np.linalg.norm(embeddings,axis=-1,keepdims=True)
            for f,emb in zip(batch,embeddings):np.save(dest/(f.stem+'.npy'),emb)
        print('FEATURES '+folder.name,flush=True)
if __name__=='__main__':main()
