from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import json
from jsonschema import validate, ValidationError
from decoder import JsonPuml

# Carga el esquema UML
with open("uml_schema.json") as f:
    UML_SCHEMA = json.load(f)

app = FastAPI()

@app.post("/uml")
async def process_uml(request: Request):
    try:
        data = await request.json()
        validate(instance=data, schema=UML_SCHEMA)  # Validación con JSON Schema
        uml_code = JsonPuml(data)          # Conversión a PlantUML
        return {"plantuml": uml_code}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")
