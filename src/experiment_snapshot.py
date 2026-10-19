"""Fingerprint image/label inputs and split membership for reproducible runs."""
import hashlib
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot(data_root, subsets, splits):
    root = Path(data_root)
    files = {}
    members = {}
    for site in subsets:
        for split in splits:
            path = root/site/'gt_sets'/(split+'.txt')
            ids = path.read_text().split()
            if not ids or len(ids)!=len(set(ids)):
                raise ValueError('Empty/duplicate split: '+str(path))
            members[site+'/'+split] = ids
            files[path.relative_to(root).as_posix()] = file_sha(path)
            for sid in ids:
                for folder,ext in [('images','.png'),('segmentation','.png'),('edge','.png'),('rebar_labels','.txt')]:
                    item = root/site/folder/(sid+ext)
                    name = item.relative_to(root).as_posix()
                    if name not in files:
                        files[name] = file_sha(item)
        path=root/site/'depth_labels.txt'
        files[path.relative_to(root).as_posix()]=file_sha(path)
        if 'train' in splits and 'val' in splits and set(members[site+'/train']) & set(members[site+'/val']):
            raise ValueError('Overlapping training and validation IDs: '+site)
    return dict(subsets=list(subsets), splits=list(splits), members=members, files=files)
