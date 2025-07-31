import os
import sys
from decoder import JsonPuml
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import jsonschema
import ollama
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

schema = {
    "type": "object",
    "properties": {
        "classes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "methods": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "inherits": {"type": "string"}
                },
                "required": ["name"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["herencia", "asociación", "agregación", "composición", "dependencia"]
                    }
                },
                "required": ["from", "to", "type"]
            }
        }
    },
    "required": ["classes"]
}


# --- Modelos de entrada ---
class PromptRequest(BaseModel):
    prompt: str

class DiagramRequest(BaseModel):
    # Puedes definir aquí el esquema esperado si lo deseas
    pass

# --- Endpoint para feedback del LLM ---
@app.post("/llm-feedback")
async def llm_feedback(data: PromptRequest):
    prompt = data.prompt
    try:
        response = ollama.chat(
            model="llama2",
            messages=[
                {"role": "system", "content": "Eres un experto en UML."},
                {"role": "user", "content": f"Este es el prompt del usuario: '{prompt}'. ¿Está bien formulado para generar un diagrama UML? Si no, sugiere cómo mejorarlo."}
            ]
        )
        return {"feedback": response['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al consultar el LLM: {e}")

# --- Endpoint para generar y enviar el diagrama ---
@app.post("/generate-diagram")
async def generate_diagram(request: Request):
    data = await request.json()
    # 1. Validación
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        return JSONResponse(status_code=400, content={"error": f"❌ Entrada no válida: {e.message}"})

    # 2. Generación del código PlantUML y del diagrama
    try:
        config = {
            "plant_uml_path": os.path.join(os.getcwd(), "plant_uml_exc"),
            "plant_uml_version": "plantuml-1.2025.2.jar",
            "json_path": None,  # No se usa aquí, pasamos el dict directamente
            "output_path": os.path.join(os.getcwd(), "output"),
            "diagram_name": "output",
        }
        json_puml = JsonPuml(config=config)
        json_puml._data = data  # Sobrescribe los datos cargados desde archivo
        json_puml._code = json_puml._json_to_plantuml()
        json_puml.generate_diagram()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error al generar el diagrama: {e}"})

    # 3. Envío del archivo generado
    svg_path = os.path.join(config["output_path"], config["diagram_name"] + ".svg")
    if not os.path.exists(svg_path):
        return JSONResponse(status_code=500, content={"error": "No se pudo generar el archivo SVG."})
    return FileResponse(svg_path, media_type="image/svg+xml")
