"""Generate image-frequency and multiscale lacunarity tables from training annotations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['lvis', 'v3det'], required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--alpha', type=float, default=1.25)
    parser.add_argument('--max-scale', type=int, default=64)
    parser.add_argument('--statistics', help='Optional original normalized-box statistics CSV')
    args = parser.parse_args()
    if args.alpha <= 0 or args.max_scale < 2:
        parser.error('alpha must be positive and max-scale must be at least two.')
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(args.annotations).read_text(encoding='utf8'))
    cats = sorted(doc['categories'], key=lambda c: c['id'])
    cids = [c['id'] for c in cats]; index = {c: i for i, c in enumerate(cids)}
    if cids != list(range(1, len(cats) + 1)):
        raise ValueError('The supplied LVIS/V3Det configurations require category IDs 1..C.')
    images = {im['id']: im for im in doc['images']}
    annotations = doc['annotations']; nclasses = len(cats)
    occurrences = [set() for _ in cats]
    for ann in annotations:
        occurrences[index[ann['category_id']]].add(ann['image_id'])
    image_counts = np.array([len(s) for s in occurrences], dtype=float)
    # This matches the tables used in the experiments, including V3Det's +1.
    smoothing = 1 if args.dataset == 'v3det' else 0
    frequency = (image_counts + smoothing) / len(images)
    if (frequency <= 0).any():
        raise ValueError('A category has zero image frequency; check dataset/category order.')
    freq = pd.DataFrame({'category_id': [0] + cids,
                         'base10': np.r_[0., -np.log10(frequency)],
                         'image_count': np.r_[0., image_counts]})
    freq.to_csv(out / f'{args.dataset}_frequency.csv', index=False)
    if args.statistics:
        stats = pd.read_csv(args.statistics)
    else:
        rows = []
        for ann in annotations:
            im = images[ann['image_id']]; x, y, w, h = ann['bbox']
            rows.append((x/im['width'], y/im['height'], w/im['width'], h/im['height'], ann['category_id']))
        stats = pd.DataFrame(rows, columns=['xmin','ymin','width','height','category'])
        # Preserve the float serialization/readback path of the original scripts.
        tmp = out / 'normalized_boxes.csv'
        stats.to_csv(tmp)
        stats = pd.read_csv(tmp)
    cx = stats.xmin.to_numpy() + stats.width.to_numpy()/2
    cy = stats.ymin.to_numpy() + stats.height.to_numpy()/2
    labels = np.array([index[int(c)] for c in stats.category], dtype=np.int64)
    counts = np.bincount(labels, minlength=nclasses)
    curves=[]
    for scale in range(1, args.max_scale+1):
        edges = np.arange(scale+1, dtype=float)*(1/scale)
        ix = np.searchsorted(edges, cx, side='right')-1
        iy = np.searchsorted(edges, cy, side='right')-1
        valid=(ix>=0)&(ix<scale)&(iy>=0)&(iy<scale)
        # Sparse cell counts reproduce the same population moments as the
        # dense S*S*C histogram, without allocating empty cells.
        key=(iy[valid]*scale+ix[valid])*nclasses+labels[valid]
        unique, mass=np.unique(key, return_counts=True)
        first=np.bincount(unique%nclasses, weights=mass, minlength=nclasses)
        second=np.bincount(unique%nclasses, weights=mass.astype(float)**2, minlength=nclasses)
        mean=first/(scale*scale)
        variance=np.maximum(second/(scale*scale)-mean**2, 0.)
        curves.append(np.log10(1 + variance**args.alpha/(mean**2+1e-6)))
    curves=np.array(curves); valid=counts>=4; slopes=np.ones(nclasses)
    def line(x, intercept, slope): return intercept+slope*x
    for c in np.flatnonzero(valid):
        upper=min(args.max_scale,int(np.sqrt(counts[c])))
        x=np.log10(np.arange(1,upper+1));y=curves[:upper,c]
        if args.dataset=='lvis':
            slopes[c]=curve_fit(line,x,y,maxfev=100000)[0][1]
        else:
            slopes[c]=np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean())
    raw=-slopes
    prior=1+(raw-raw.min())/np.ptp(raw) if np.ptp(raw)>0 else np.ones(nclasses)
    prior[~valid]=1.
    names=[c['name'] for c in cats]
    pd.DataFrame({'category_id':cids,'class_name':names,'lambda':prior}).to_csv(
        out/f'{args.dataset}_lacunarity.csv',index=False,float_format='%.17g')
    print(f'Wrote {nclasses} category priors to {out}; alpha={args.alpha}, cap={args.max_scale}.')


if __name__ == '__main__':
    main()
