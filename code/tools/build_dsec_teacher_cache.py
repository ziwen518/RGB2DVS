"""Build paired RGB/event DINO caches for complete DSEC sequences."""
from __future__ import annotations
import argparse, io, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "core"))
from data import make_event_dataset  # noqa: E402
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def teacher(device):
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    return torch.hub.load(str(repo), "dinov2_vits14", source="local").eval().to(device)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--split',default='train'); p.add_argument('--steps',type=int,default=16); p.add_argument('--size',type=int,default=48); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--gpu',type=int,default=1); p.add_argument('--max-samples',type=int)
    a=p.parse_args(); device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    ds=make_event_dataset('dsec',a.root, a.split=='train', steps=a.steps,size=a.size,max_samples=a.max_samples)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0)
    net=teacher(device); prefixes=[]; patches=[]; rgb_prefixes=[]; event_patches=[]; ids=[]
    image_cache={}
    @torch.no_grad()
    def rgb_batch(batch_ids):
        images=[]
        for sid in batch_ids:
            _,seq,frame=sid.split('/'); frame_zip=a.root/a.split/f'{seq}/{seq}_images_rectified_left.zip'
            key=(str(frame_zip),frame)
            if key not in image_cache:
                with zipfile.ZipFile(frame_zip) as z: image_cache[key]=z.read(f'{frame}.png')
            arr=np.asarray(Image.open(io.BytesIO(image_cache[key])).convert('RGB'),dtype=np.float32)/255.0
            images.append(torch.from_numpy(arr).permute(2,0,1))
        x=F.interpolate(torch.stack(images).to(device),size=(224,224),mode='bicubic',align_corners=False); x=(x-MEAN.to(device))/STD.to(device)
        out=net.forward_features(x); return F.normalize(out['x_norm_clstoken'].float(),dim=-1).cpu(), F.normalize(F.adaptive_avg_pool2d(out['x_norm_patchtokens'].transpose(1,2).reshape(len(images),384,16,16),(8,8)).flatten(2).transpose(1,2).float(),dim=-1).cpu()
    with torch.no_grad():
      for events,_,_,sid in dl:
        events=events.to(device); pref=[]
        ep=[]
        for t in range(events.shape[1]):
          # Match the existing event teacher: cumulative temporal prefix,
          # then polarity-average to one event-derived grayscale image.
          x=events[:,:t+1].sum(1).mean(1,keepdim=True).expand(-1,3,-1,-1); x=F.interpolate(x,size=(224,224),mode='bicubic',align_corners=False); x=(x-MEAN.to(device))/STD.to(device); out=net.forward_features(x); pref.append(F.normalize(out['x_norm_clstoken'].float(),dim=-1).cpu()); ep.append(F.normalize(F.adaptive_avg_pool2d(out['x_norm_patchtokens'].transpose(1,2).reshape(len(x),384,16,16),(8,8)).flatten(2).transpose(1,2).float(),dim=-1).cpu())
        rp,rt=rgb_batch(list(sid)); prefixes.append(torch.stack(pref,1)); event_patches.append(torch.stack(ep,1)); rgb_prefixes.append(rp); patches.append(rt); ids.extend(map(str,sid))
        print(json.dumps({'samples':len(ids)}),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True); torch.save({'ids':ids,'prefix':torch.cat(prefixes),'patches':torch.cat(patches),'rgb_prefix':torch.cat(rgb_prefixes),'event_patches':torch.cat(event_patches),'labels':torch.zeros(len(ids),dtype=torch.long)},a.out)
    print(json.dumps({'out':str(a.out),'samples':len(ids),'device':str(device)}),flush=True)
if __name__=='__main__': main()
