"""Validate shipped split membership, image pairs and physical references."""
import json
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inspection_data import AnnotationStore
from measure import load_depth_labels


def main():
    store = AnnotationStore(ROOT / 'annotation_all_images/all_images_annotations.csv')
    result = {}
    for site in ('site-1', 'site-2'):
        base = ROOT / 'data' / site
        splits = {s: (base / 'gt_sets' / (s + '.txt')).read_text(encoding='utf-8').split() for s in ('train', 'val')}
        assert not set(splits['train']) & set(splits['val']), 'Train/val overlap'
        ranges = load_depth_labels(str(base / 'depth_labels.txt'))
        for split, ids in splits.items():
            assert len(ids) == len(set(ids)), 'Duplicate IDs'
            for sid in ids:
                for folder in ('images', 'segmentation', 'edge'):
                    path = base / folder / (sid + '.png')
                    with Image.open(path) as im:
                        im.verify()
                xs = (base / 'rebar_labels' / (sid + '.txt')).read_text(encoding='utf-8').splitlines()
                assert len([x for x in xs if x.strip()]) - 1 == len(store.spacings(site, sid)), site + '/' + sid
                assert ranges[sid] > 0
            result[site + '/' + split] = len(ids)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
