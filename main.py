import os
import sys
from decoder import JsonPuml
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import jsonschema
from fastapi.middleware.cors import CORSMiddleware
import requests
import ollama

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# --- Endpoint para recibir feedback del LLM ---
@app.post("/api/llm-feedback")
async def llm_feedback(data: PromptRequest):
    try:
        system_prompt = (
            "Responde siempre en español. Eres un experto en ingeniería de software y diagramas UML. "
            "Tu tarea es revisar si los enunciados están bien redactados para generar un diagrama UML adecuado."
        )
        user_prompt = (
            f"Este es el prompt del usuario: '{data.prompt}'. ¿Está bien formulado para generar un diagrama UML? "
            "Si no, sugiere cómo mejorarlo en respuestas breves que tengan entre 100 y 300 palabras."
        )

        full_prompt = f"{system_prompt}\n{user_prompt}"

        # Función generadora para enviar el texto a medida que llega
        def stream_feedback():
            for chunk in ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": full_prompt}],
                stream=True
            ):
                yield chunk["message"]["content"]

        return StreamingResponse(stream_feedback(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {e}")
    
# --- Endpoint para generar y enviar el diagrama ---
@app.post("/generate-diagram")
async def generate_diagram(request: Request):
    data = await request.json()

    # 1️⃣ Validación con jsonschema
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"❌ Entrada no válida: {e.message}"}
        )

    # Check if we should stream the response or generate a diagram
    if "stream" in data and data["stream"]:
        # 2️⃣ Función generadora para streaming
        def stream_response():
            for chunk in ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": data["prompt"]}],
                stream=True
            ):
                # Enviar solo el texto del mensaje
                yield chunk["message"]["content"]

        # 3️⃣ Devolver streaming al frontend
        return StreamingResponse(stream_response(), media_type="text/plain")
    else:
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
