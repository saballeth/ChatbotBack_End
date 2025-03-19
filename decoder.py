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

def json_to_plantuml(data:dict) -> str:
    plantuml_str =  "@startuml Diagram\n"

    if data["diagramType"] == "classDiagram":
        plantuml_str += decodeClassDiagram(data)
    elif data["diagramType"] == "useCaseDiagram":
        plantuml_str += decodeUseCaseDiagram(data)

    plantuml_str += "@enduml"
    return plantuml_str

def decodeClassDiagram(data:dict) -> str:
    plantuml_str = ""

    # class
    for element in data.get('declaringElements', []):
        # declare the class
        type = element.get('type', '')
        elementName = element.get('name', '')
        plantuml_str += f"{type} {elementName}\n"
        
        # adding attributes for clases
        for attribute in element.get('attributes', []):
            attributeName = attribute.get('name', '')
            attributeType = attribute.get('type', '')
            visibility = visibilities[attribute.get('visibility', '')]
            static = "{isStatic}" if attribute.get('isStatic', False) else ''
            final = "{final}" if attribute.get('isFinal', False) else ''
            plantuml_str += f"{elementName} : {visibility} {static} {final} {attributeType} {attributeName}\n"

        # adding methods for clases    
        for method in element.get('methods', []):
            methodName = method.get('name', '')
            abstract = "{abstract}" if method.get('isAbstract', False) else ''
            visibility = visibilities[method.get('visibility', 'public')]
            returnType =  method.get('returnType', 'void')
            
            # TODO add params (make a for to params inside the method)
            params = ""

            plantuml_str += f"{elementName} : {visibility} {abstract} {returnType} {methodName}({params})\n"
        
    # adding relationsihps
    for relation in data.get('relationShips', []):
        relationType = relations_type[relation.get('type', '')]
        source = relation.get('source', '') 
        target = relation.get('target', '')
        
        # TODO add multiplicity 
        multiplicity = f"{relation['multiplicity']} " if 'multiplicity' in relation else '1'
        multiplicityEnd1 = multiplicity # split the multiplicity to end 1
        multiplicityEnd2 = multiplicity # split the multiplicity to end 1
        plantuml_str += f"{source} \"{multiplicityEnd1}\" {relationType} \"{multiplicityEnd2}\" {target}\n"

    return plantuml_str


def decodeUseCaseDiagram(data:dict):
    plantuml_str = ""

    # Actores
    for actor in data.get("actors", []):
        plantuml_str = decodeUseCaseActor(plantuml_str, actor)
        
    # Casos de uso globales
    for use_case in data.get("useCases", []):
        plantuml_str = decodeUseCase(plantuml_str, use_case)

    # Paquetes
    plantuml_str = decodeUseCasePackage(plantuml_str, data)

    # Relaciones
    for relation in data.get("relationships", []):
        plantuml_str = decodeRelationships(plantuml_str, relation)     

    return plantuml_str


def decodeUseCaseActor(current_code:str, data) -> str:
    plantuml_str = current_code

    actor_name = data.get("name", "")
    actor_alias = data.get("alias", "") 
    actor_stereotype = f' <<{data["stereotype"]}>>' if "stereotype" in data else ""
    actor_business = "/" if data.get("business", False) else ""

    if actor_name:
        plantuml_str += f"actor{actor_business} \"{actor_name}\""
        if actor_alias:
            plantuml_str += f" as {actor_alias}"
        plantuml_str += actor_stereotype + "\n"
    
    return plantuml_str

def decodeUseCase(current_code:str, data) -> str:
    plantuml_str = current_code

    usecase_name = data.get("name", "")
    if usecase_name:
        usecase_business = "/" if data.get("business", False) else ""
        usecase_alias = data.get("alias", "")
        usecase_stereotype = actor_stereotype = f' <<{data["stereotype"]}>>' if "stereotype" in data else ""
        if usecase_alias:
            plantuml_str += f'usecase{usecase_business} (\"{usecase_name}\") as {usecase_alias} {usecase_stereotype}\n'
   
    return plantuml_str


def decodeUseCasePackage(current_code: str, data) -> str:
    plantuml_str = current_code
    packages = data.get("packages", [])
    if not packages:
        return plantuml_str
    
    for package in packages:
        plantuml_str += f'package "{package["name"]}" as {package["alias"]} {{\n'

        for use_case in package.get("useCases", []):
            plantuml_str = decodeUseCase(plantuml_str, use_case) 

        for actor in package.get("actors", []):
            plantuml_str = decodeUseCaseActor(plantuml_str, actor)

        plantuml_str = decodeUseCasePackage(plantuml_str, package)

        plantuml_str += "}\n" 

    return plantuml_str
def decodeRelationships(current_code:str, data) -> str:
    plantuml_str = current_code

    relation_type = data.get("type", "")
    relation_principal = data.get("principal", "")
    relation_secondary = data.get("secondary", "")
    relation_direction = data.get("direction", "")
    relation_label = f":\"{data.get('label', '')}\"" if data.get('label') else ""

    #Only actor_actor
    relation_extend = data.get("extend", "")
    
    #only usecase_usecase 
    relation_stereotype = data.get("stereotype", "")

    if(relation_principal and relation_secondary):
        if relation_type == "actor_actor": 
            if relation_extend == ">":
                plantuml_str += f"{relation_secondary} <|-- {relation_principal}"
            elif relation_extend == "<":
                plantuml_str += f"{relation_principal} <|-- {relation_secondary}"
            else:
                plantuml_str += f"{relation_principal} -{relation_direction}-> {relation_secondary}"
            plantuml_str += relation_label + "\n"
                
        elif relation_type == "actor_usecase":
            plantuml_str += f"{relation_principal} -{relation_direction}-> {relation_secondary}{relation_label}\n"

        elif relation_type == "useCase_usecase":
            if relation_stereotype == "include" :
                plantuml_str += f"{relation_principal} .> {relation_secondary}: include\n"
            elif relation_stereotype == "extends":
                plantuml_str += f"{relation_principal} .> {relation_secondary}: extends\n"

        elif relation_type == "package_package":
            plantuml_str += f"{relation_principal} -{relation_direction}-> {relation_secondary}{relation_label}\n"

    return plantuml_str


        

