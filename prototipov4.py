import json
import sys


def cargar_json_desde_archivo():
    """Carga un archivo JSON desde una ruta fija y lo devuelve como un diccionario."""
    with open(sys.argv[1], "r", encoding="utf-8") as input:
        return json.load(input)


def json_a_plantuml(json_data):
    plantuml_code = "@startuml Diagram\n"

    # Determinar el tipo de diagrama
    diagram_type = json_data.get("diagramType", "classDiagram")
    if diagram_type == "useCaseDiagram":
        plantuml_code += generar_use_case_diagram(json_data)
    else:
        plantuml_code += generar_class_diagram(json_data)

    plantuml_code += "@enduml"
    return plantuml_code

def generar_class_diagram(json_data):
    plantuml_code = ""

    # Generar clases
    for clase in json_data.get("classes", []):
        plantuml_code += f"class {clase['name']} {{\n"
        
        # Atributos
        for atributo in clase.get("attributes", []):
            visibility = {"public": "+", "private": "-", "protected": "#"}.get(atributo["visibility"], "-")
            static = "{static}" if atributo.get("isStatic", False) else ""
            final = "{final}" if atributo.get("isFinal", False) else ""
            plantuml_code += f"  {visibility} {atributo['name']}: {atributo['type']} {static} {final}\n"
        
        # Métodos
        for metodo in clase.get("methods", []):
            visibility = {"public": "+", "private": "-", "protected": "#"}.get(metodo.get("visibility", "public"), "+")
            parametros = ", ".join(metodo.get("parameters", []))
            abstract = "{abstract}" if metodo.get("isAbstract", False) else ""
            plantuml_code += f"  {visibility} {metodo['name']}({parametros}): {metodo.get('returnType', 'void')} {abstract}\n"
        
        plantuml_code += "}\n"

        # Relaciones
        for relacion in clase.get("relationships", []):
            tipo = {
                "inheritance": "<|--",
                "composition": "*--",
                "aggregation": "o--",
                "association": "--",
                "instantiation": "..|>",
                "realization": "<|.."
            }.get(relacion["type"], "--")
            multiplicidad = f' "{relacion["multiplicity"]}" ' if "multiplicity" in relacion else ""
            plantuml_code += f"{clase['name']} {tipo} {multiplicidad}{relacion['target']}\n"

        # Interfaces implementadas
        for interfaz in clase.get("implementsInterfaces", []):
            plantuml_code += f"{clase['name']} ..|> {interfaz}\n"

    # Generar interfaces
    for interfaz in json_data.get("interfaces", []):
        plantuml_code += f"interface {interfaz['name']} {{\n"
        for metodo in interfaz.get("methods", []):
            plantuml_code += f"  + {metodo['name']}(): {metodo.get('returnType', 'void')}\n"
        plantuml_code += "}\n"

    return plantuml_code

def generar_use_case_diagram(json_data):
    plantuml_code = ""

    # Generar actores
    for actor in json_data.get("actors", []):
        tipo = actor.get("type", "primary")
        plantuml_code += f"actor {actor['name']} as {actor['name']} {'<<secondary>>' if tipo == 'secondary' else ''}\n"

    # Generar casos de uso
    for use_case in json_data.get("useCases", []):
        plantuml_code += f"usecase {use_case['name']} as {use_case['name']}\n"

        # Relaciones con actores
        for actor in use_case.get("actors", []):
            plantuml_code += f"{actor} --> {use_case['name']}\n"

        # Relaciones con otros casos de uso
        for relacion in use_case.get("relationships", []):
            tipo = {
                "include": "include",
                "extend": "extend",
                "association": "--"
            }.get(relacion["type"], "--")
            if(tipo == "include"):
                plantuml_code += f"{use_case['name']} -> {relacion['target']} : <<{tipo}>>\n"
            elif(tipo == "extend"):
                plantuml_code += f"{use_case['name']} .> {relacion['target']} : <<{tipo}>>\n"
            else:
                plantuml_code += f"{use_case['name']} {tipo} {relacion['target']}\n"
        
    return plantuml_code


def main():
    # Cargar el archivo JSON
    json_data = cargar_json_desde_archivo()

    # Generar el código PlantUML
    plantuml_code = json_a_plantuml(json_data)

    with open("output.puml", "+w") as output:
        output.write(plantuml_code)

    # Imprimir el resultado
    print(plantuml_code)

# Ejecutar el script inmediatamente
main()