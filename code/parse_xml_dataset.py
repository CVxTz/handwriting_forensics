import json
import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

from tqdm import tqdm

base_path = Path("../input/")
jsons_path = base_path / "jsons"
os.makedirs(jsons_path, exist_ok=True
            )
texts = list(glob(str(base_path / "**/*.txt"), recursive=True))

for text_path in tqdm(texts):
    with open(text_path, 'r') as f:
        lines = f.read().split('CSR:')[-1].strip().split("\n")
    stem = str(Path(text_path).stem)
    xmls = sorted(list(glob(str(base_path / ("**/%s*.xml" % stem)), recursive=True)))

    if len(xmls) == len(lines):
        for (xml, line) in zip(xmls, lines):
            xml_stem = Path(xml).stem
            sample = {"text": line}
            root = ET.parse(xml).getroot()
            all_points = root.findall('.//Point')

            min_x = min([int(p.get("x")) for p in all_points])
            max_x = max([int(p.get("x")) for p in all_points])

            min_y = min([int(p.get("y")) for p in all_points])
            max_y = max([int(p.get("y")) for p in all_points])

            sample['max_x'] = max_x - min_x
            sample['max_y'] = max_y - min_y

            list_strokes = []

            strokes = root.findall('StrokeSet')[0].findall('Stroke')

            for stroke in strokes:
                points = stroke.findall('Point')
                list_strokes.append(
                    {"x": [int(p.get("x")) - min_x for p in points], "y": [int(p.get("y")) - min_y for p in points]})
            sample['strokes'] = list_strokes

            with open(jsons_path / (xml_stem + ".json"), "w") as f:
                json.dump(sample, f, indent=4)