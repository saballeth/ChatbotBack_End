
import json
import os
import subprocess

class JsonPuml:
    def __init__(self, config: dict):

        self._plant_uml_path = config['plant_uml_path']
        self._plant_uml_version = config['plant_uml_version']
        self._json_path = config['json_path']
        self._output_path = config['output_path']
        self._diagram_name = config['diagram_name']

        self._data = self._get_data()
        self._code = self._json_to_plantuml()

    def generate_diagram(self):
        os.makedirs(self._output_path, exist_ok=True)
        out = os.path.join(self._output_path, self._diagram_name+".puml")
        plant_uml = os.path.join(self._plant_uml_path, self._plant_uml_version)

        with open(out, "+w", encoding="utf-8") as output:
            output.write(self._code)
    
        subprocess.run([
            "java", "-jar", plant_uml,            
            "-o", self._output_path,     
            out                           
        ])
        

    
    def _get_data(self) -> dict:
        with open(self._json_path, "r", encoding="utf-8") as input:
            data = json.load(input) 
        return data
    
    def _json_to_plantuml(self) -> str:
        plantuml_str =  "@startuml Diagram\n"

        if self._data["diagramType"] == "classDiagram":
            decode_class = DecodeClass(self._data)
            plantuml_str += decode_class.get_code()
        elif self._data["diagramType"] == "useCaseDiagram":
            decode_use_case = DecodeUseCase(self._data)
            plantuml_str += decode_use_case.get_code()

        plantuml_str += "@enduml"
        return plantuml_str
    

class DecodeUseCase:
    def __init__(self, data:dict):
        self._data = data
        self._use_case_code = self._generate_code()

    def get_code(self) -> str:
        return self._use_case_code

    def _generate_code(self) -> str:
        plantuml_str = ""

        # Actores
        for actor in self._data.get("actors", []):
            plantuml_str = self._decodeUseCaseActor(plantuml_str, actor)
            
        # Casos de uso globales
        for use_case in self._data.get("useCases", []):
            plantuml_str = self._decodeUseCase(plantuml_str, use_case)

        # Paquetes
        plantuml_str = self._decodeUseCasePackage(plantuml_str, self._data)

        # Relaciones
        for relation in self._data.get("relationships", []):
            plantuml_str = self._decodeRelationships(plantuml_str, relation)     

        return plantuml_str


    def _decodeUseCaseActor(self, current_code:str, data) -> str:
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

    def _decodeUseCase(self, current_code:str, data) -> str:
        plantuml_str = current_code

        usecase_name = data.get("name", "")
        if usecase_name:
            usecase_business = "/" if data.get("business", False) else ""
            usecase_alias = data.get("alias", "")
            usecase_stereotype = actor_stereotype = f' <<{data["stereotype"]}>>' if "stereotype" in data else ""
            if usecase_alias:
                plantuml_str += f'usecase{usecase_business} (\"{usecase_name}\") as {usecase_alias} {usecase_stereotype}\n'
    
        return plantuml_str


    def _decodeUseCasePackage(self, current_code: str, data) -> str:
        plantuml_str = current_code
        packages = data.get("packages", [])
        if not packages:
            return plantuml_str
        
        for package in packages:
            plantuml_str += f'package "{package["name"]}" as {package["alias"]} {{\n'

            for use_case in package.get("useCases", []):
                plantuml_str = self._decodeUseCase(plantuml_str, use_case) 

            for actor in package.get("actors", []):
                plantuml_str = self._decodeUseCaseActor(plantuml_str, actor)

            plantuml_str = self._decodeUseCasePackage(plantuml_str, package)

            plantuml_str += "}\n" 

        return plantuml_str
    
    def _decodeRelationships(self, current_code:str, data) -> str:
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



class DecodeClass:
    def __init__(self, data:dict):

        self._data = data
        self._class_code = self._generate_code()
        
        self._visibilities = {
        "private": "-",
        "protected": "#",
        "package private": "~",
        "public": "+"
        }
        self._relations_type = {
            "inheritance": "<|--",
            "composition": "*--",
            "aggregation": "o--",
            "association": "--",
            "instantiation": "..|>",
            "realization": "<|..",
        }
        
    def get_code(self) -> str:
        return self._class_code

    def _generate_code(self) -> str:
        plantuml_str = ""
        for element in self._data.get('declaringElements', []):
            # declare the class
            type = element.get('type', '')
            elementName = element.get('name', '')
            plantuml_str += f"{type} {elementName}\n"
            
            # adding attributes for clases
            for attribute in element.get('attributes', []):
                attributeName = attribute.get('name', '')
                attributeType = attribute.get('type', '')
                visibility = self._visibilities[attribute.get('visibility', '')]
                static = "{isStatic}" if attribute.get('isStatic', False) else ''
                final = "{final}" if attribute.get('isFinal', False) else ''
                plantuml_str += f"{elementName} : {visibility} {static} {final} {attributeType} {attributeName}\n"

            # adding methods for clases    
            for method in element.get('methods', []):
                methodName = method.get('name', '')
                abstract = "{abstract}" if method.get('isAbstract', False) else ''
                visibility = self._visibilities[method.get('visibility', 'public')]
                returnType =  method.get('returnType', 'void')
                # TODO add params (make a for to params inside the method)
                params = ""
                for param in method.get('params', []):
                    params += f"{param.get('type', '')} {param.get('name', '')} "

                plantuml_str += f"{elementName} : {visibility} {abstract} {returnType} {methodName}({params[:-1]})\n" # params[:-1] para quitar un espcio en blanco
            
            # adding relationsihps
            for relation in self._data.get('relationShips', []):
                relationType = self._relations_type[relation.get('type', '')]
                source = relation.get('source', '') 
                target = relation.get('target', '')
                multiplicityEnd1 = f"\"{relation['multiplicity'][0]}\"" if 'multiplicity' in relation else ''
                multiplicityEnd2 = f"\"{relation['multiplicity'][3]}\"" if 'multiplicity' in relation else ''
                
                plantuml_str += f"{source} {multiplicityEnd1} {relationType} {multiplicityEnd2} {target}\n"

        return plantuml_str
        

