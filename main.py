import os
import sys
from decoder import *

plant_uml_path = os.path.join(os.getcwd(), "plant_uml_exc")
plant_uml_version = "plantuml-1.2025.2.jar"
json_path = os.path.join(os.getcwd(), "inputs", sys.argv[1])
output_path = os.path.join(os.getcwd(), "output")
diagram_name = "output"


config = {
    "plant_uml_path": plant_uml_path,
    "plant_uml_version": plant_uml_version,
    "json_path": json_path,
    "output_path": output_path,
    "diagram_name": diagram_name,
}

json_puml = JsonPuml(config = config)
json_puml.generate_diagram()
