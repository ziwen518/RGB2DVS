"""Create deterministic label-free train/validation IDs from DSEC frame IDs."""
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'core'))
from data import DSECEvents

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--validation-fraction',type=float,default=.1); p.add_argument('--max-samples',type=int); a=p.parse_args()
 ds=DSECEvents(a.root,True,max_samples=a.max_samples); ids=[f'train/{s[0]}/{s[1]:06d}' for s in ds.samples]
 by={}
 for sid in ids: by.setdefault(sid.split('/')[0],[]).append(sid)
 train=[]; val=[]
 for seq, xs in sorted(by.items()):
  n=max(1,int(len(xs)*a.validation_fraction)); val.extend(xs[:n]); train.extend(xs[n:])
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({'train_ids':train,'validation_ids':val,'test_ids':[],'source':'DSEC complete frame manifest; labels unused'},indent=2),encoding='utf-8'); print({'train':len(train),'validation':len(val)})
if __name__=='__main__': main()
