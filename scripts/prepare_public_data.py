"""Convert the Chinese-named public subset into the existing runner layout."""
import csv
import shutil
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    data, ann = ROOT / 'data', ROOT / 'annotation_all_images'
    if data.exists() or ann.exists():
        raise FileExistsError('data/ or annotation_all_images/ exists; refusing to overwrite')
    public = ROOT / 'public_data'
    with (public / 'samples.csv').open(encoding='utf-8-sig', newline='') as f:
        samples = list(csv.DictReader(f))
    with (public / 'measurements.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    by_id = defaultdict(list)
    for r in records:
        by_id[r['sample_id']].append(r)
    converted = []
    for number, name in ((1, '工地一'), (2, '工地二')):
        site = f'site-{number}'
        base = data / site
        for folder in ('images', 'segmentation', 'edge', 'rebar_labels', 'gt_sets'):
            (base / folder).mkdir(parents=True)
        chosen = [s for s in samples if s['site'] == name]
        for s in chosen:
            sid = s['sample_id']
            with Image.open(public / s['image']) as im:
                im.convert('RGB').save(base / 'images' / (sid + '.png'))
            for folder, field, suffix in (('segmentation', 'segmentation', '.png'), ('edge', 'edge', '.png'), ('rebar_labels', 'centers', '.txt')):
                shutil.copy2(public / s[field], base / folder / (sid + suffix))
            for r in by_id[sid]:
                row = dict(r, sample_key=f'{site}/{sid}', subset=site, image_id=sid,
                           working_spacing_cm=str(Decimal(r['submitted_spacing_mm']) / 10),
                           split=s['original_split'], image_path=f'data/{site}/images/{sid}.png')
                converted.append(row)
        for split in ('train', 'val', 'unassigned'):
            (base / 'gt_sets' / (split + '.txt')).write_text(''.join(s['sample_id'] + '\n' for s in chosen if s['original_split'] == split), encoding='utf-8')
        (base / 'depth_labels.txt').write_text(''.join(f"{s['sample_id']}\t{s['laser_distance_m']}\n" for s in chosen), encoding='utf-8')
    ann.mkdir()
    with (ann / 'all_images_annotations.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    print(f'Prepared {len(samples)} public images and {len(converted)} intervals. Original splits retained.')


if __name__ == '__main__':
    main()
