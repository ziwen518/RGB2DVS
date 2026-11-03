"""Build cached [T,2,H,W] frames for extracted N-ImageNet npz events."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--steps',type=int,default=16); p.add_argument('--size',type=int,default=48); args=p.parse_args()
    import sys; sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
    from data import events_to_frames
    paths=sorted((args.root/'events').rglob('*.npz')); out=args.root/f'frames_nimagenet_t{args.steps}_s{args.size}'
    for i,path in enumerate(paths,1):
        rel=path.relative_to(args.root/'events').with_suffix('.npy'); dst=out/rel
        if dst.exists(): continue
        with np.load(path,allow_pickle=False) as payload:
            source = payload['event_data'] if 'event_data' in payload.files else payload
            event=np.zeros(len(source),dtype=[('x',np.int16),('y',np.int16),('t',np.int64),('p',np.uint8)])
            for field in event.dtype.names: event[field]=source[field]
        frames=events_to_frames(event,args.steps,args.size,'count_global')
        dst.parent.mkdir(parents=True,exist_ok=True); np.save(dst,frames.mul(255).round().to(torch.uint8).numpy(),allow_pickle=False)
        if i==1 or i%1000==0: print({'processed':i,'total':len(paths)},flush=True)
    print({'frames':len(paths),'output':str(out)},flush=True)
if __name__=='__main__':
    import torch
    main()
