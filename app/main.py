import locale
import nest_asyncio
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from models.llama import LlamaModel
from models.langchain import ConversationChain
from utils.text_processing import wrap_text_preserve_newlines

locale.getpreferredencoding = lambda: "UTF-8"

app = FastAPI()

# Initialize models
llama_model = LlamaModel()
conversation = ConversationChain(llm=llama_model.pipeline)

# Define API routes
@app.post("/reset")
async def reset():
    conversation.reset_memory()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    response = conversation.predict(input=data['input_text'])
    response = response.split("Human:")
    response = response[0]
    response = wrap_text_preserve_newlines(response)
    data = {
        'response': response
    }
    return JSONResponse(data)

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)


