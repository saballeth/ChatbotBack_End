import sys
import json
import decoder


with open("./inputs/" + sys.argv[1], "r", encoding="utf-8") as input:
    data = json.load(input)

plant_uml_code = decoder.json_to_plantuml(data)

with open("output.puml", "+w", encoding="utf-8") as output:
    output.write(plant_uml_code)