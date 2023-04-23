from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

class Langchain:
    def __init__(self, model_name_or_path):
        self.llm = HuggingFacePipeline(
            pipeline="text-generation",
            model=model_name_or_path,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=4)
        )

    def predict(self, input_text):
        response = self.conversation.predict(input=input_text)
        response = response.split("Human:")
        response = response[0]
        return response
