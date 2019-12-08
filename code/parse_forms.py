from collections import defaultdict
import json
from glob import glob
from pathlib import Path


if __name__=="__main__":

    with open('../input/forms.txt', 'r') as f:
        lines = f.read().split("\n")

    lines = [x.split() for x in lines if not x.startswith("#") and x]

    writer_form_mapping = defaultdict(list)
    writer_images_mapping = defaultdict(list)
    writer_json_mapping = defaultdict(list)

    form_writer_mapping = {}

    for line in lines:
        writer_form_mapping[line[1]].append(line[0])
        form_writer_mapping[line[0]] = line[1]

    with open('writer_form_mapping.json', "w") as f:
        json.dump(writer_form_mapping, f, indent=4)

    image_paths = glob("../input/clean_images/**/*.png", recursive=True)
    json_paths = glob("../input/jsons/*.json")

    for path in image_paths:
        name = Path(path).stem
        base_name = "-".join(name.split("-")[:2])
        if base_name in form_writer_mapping:
            writer_images_mapping[form_writer_mapping[base_name]].append(path)

    with open('writer_images_mapping.json', "w") as f:
        json.dump(writer_images_mapping, f, indent=4)

    for path in json_paths:
        name = Path(path).stem
        base_name = "-".join(name.split("-")[:2])
        if base_name in form_writer_mapping:
            writer_json_mapping[form_writer_mapping[base_name]].append(path)

    with open('writer_json_mapping.json', "w") as f:
        json.dump(writer_json_mapping, f, indent=4)


