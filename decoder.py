def json_to_plantuml(data:dict) -> str:
    plantuml_str =  "@startuml Diagram\n"

    if data["diagramType"] == "classDiagram":
        plantuml_str += decodeClassDiagram(data)
    elif data["diagramType"] == "useCaseDiagram":
        plantuml_str += decodeUseCaseDiagram(data)

    plantuml_str += "@enduml"
    return plantuml_str

def decodeClassDiagram(data:dict) -> str:
    visibilities = {
        "private": "-",
        "protected": "#",
        "package private": "~",
        "public": "+"
    }
    relations_type = {
        "inheritance": "<|--",
        "composition": "*--",
        "aggregation": "o--",
        "association": "--",
        "instantiation": "..|>",
        "realization": "<|..",
    }

    isStatic = '{static} '
    plantuml_str = ""
    # class
    for clase in data.get("classes", []):
        # declare the class
        plantuml_str += f"{'abstract ' if clase.get('isAbstract', '') else ''}class {clase['name']}\n"
        
        # adding attributes for clases
        for attribute in clase.get("attributes", []):
            visibility = visibilities[attribute.get('visibility', '')]
            static = "{isStatic}" if attribute['isStatic'] else ''
            final = "{final}" if attribute.get('isFinal', False) else ''
            plantuml_str += f"{clase['name']} : {visibility} {static} {final} {attribute['name']} {attribute['type']}\n"

        # adding methods for clases    
        for method in clase.get('methods', []):
            abstract = "{abstract}" if method.get('isAbstract', False) else ''
            visibility = visibilities[method.get('visibility', 'public')]
            returnType =  method.get('returnType', 'void')
            plantuml_str += f"{clase.get('name', '')} : {visibility} {abstract} {returnType} {method.get('name', '')}()\n"
        
        # adding relationsihps
        for relation in clase.get('relationships', []):
            relationType = relations_type[relation.get('type', [])]
            # to implement the muliplicity
            multiplicity = f"{relation['multiplicity']} " if 'multiplicity' in relation else ''
            plantuml_str += f"{clase['name']} {relationType} {relation['target']}\n"

        # adding interfaces implementeds
        for interfaz in clase.get('implementsInterfaces', []):
            plantuml_str += f"{clase['name']} ..|> {interfaz}\n"
    
    # interfaces
    for interfaz in data.get('interfaces', []):
        plantuml_str += f"interface {interfaz.get('name', 'interfazName')}\n"
        for method in interfaz.get('methods', []):
            plantuml_str += f"{interfaz.get('name', 'interfazName')} : {method.get('returnType', 'void')} {method.get('name', '')}()\n"
    
    return plantuml_str

# to do
def decodeUseCaseDiagram(data:dict):
    return 'plantuml to generate'
