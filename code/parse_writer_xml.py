import json
import xml.etree.ElementTree as ET


if __name__=="__main__":

    xml = '../input/writers.xml'

    root = ET.parse(xml).getroot()
    all_writers = root.findall('.//Writer')

    writer_gender_mapping = {x.get("name"): 1 if x.get("Gender") == "Male" else 0 for x in all_writers}

    with open('writer_gender_mapping.json', "w") as f:
        json.dump(writer_gender_mapping, f, indent=4)


