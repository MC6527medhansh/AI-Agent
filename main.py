import ast
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template
from code.code_reader import code_reader

# Load environment variables
load_dotenv()

class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

def initialize_llms():
    print("Initializing LLMs...")
    return Ollama(model="mistral", request_timeout=30.0), Ollama(model="codellama")

def initialize_parser():
    print("Initializing parser...")
    return LlamaParse(result_type="markdown")

def load_documents(parser):
    print("Loading documents...")
    file_extractor = {".pdf": parser}
    return SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

def create_vector_index(documents):
    print("Creating vector index...")
    embed_model = resolve_embed_model("local:BAAI/bge-m3")
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

def create_query_engine(vector_index, llm):
    print("Creating query engine...")
    return vector_index.as_query_engine(llm=llm)

def setup_tools(query_engine):
    print("Setting up tools...")
    return [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="api_documentation",
                description="this gives documentation about code for an API. Use this for reading docs for the API."
            ),
        ),
        code_reader,
    ]

def create_agent(tools, code_llm):
    print("Creating agent...")
    return ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

def setup_output_pipeline():
    print("Setting up output pipeline...")
    parser = PydanticOutputParser(CodeOutput)
    json_prompt_str = parser.format(code_parser_template)
    json_prompt_tmpl = PromptTemplate(json_prompt_str)
    return QueryPipeline(chain=[json_prompt_tmpl, llm])

def main():
    llm, code_llm = initialize_llms()
    parser = initialize_parser()
    documents = load_documents(parser)
    vector_index = create_vector_index(documents)
    query_engine = create_query_engine(vector_index, llm)
    tools = setup_tools(query_engine)
    agent = create_agent(tools, code_llm)
    output_pipeline = setup_output_pipeline()
    
    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        retries = 0
        while retries < 3:
            try:
                print(f"Processing prompt: {prompt}")
                result = agent.query(prompt)
                print("Result from agent query:", result)
                next_result = output_pipeline.run(response=result)
                print("Next result from output pipeline:", next_result)
                cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
                break
            except Exception as e:
                retries += 1
                print(f"Error occurred, retry #{retries}:", e)
        
        if retries >= 3:
            print("Unable to process request, try again...")
            continue
        
        print("Code generated")
        print(cleaned_json["code"])
        print("\n\nDescription:", cleaned_json["description"])
        
        filename = cleaned_json["filename"]
        try:
            with open(os.path.join("output", filename), "w") as f:
                f.write(cleaned_json["code"])
            print("Saved file", filename)
        except Exception as e:
            print("Error saving file:", e)

if __name__ == "__main__":
    main()
