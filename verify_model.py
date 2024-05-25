from ollama import Ollama
import os

def verify_ollama_model(model_name):
    try:
        print(f"Initializing Ollama model: {model_name}...")
        llm = Ollama(model=model_name, request_timeout=30.0)
        print(f"Model {model_name} initialized successfully.")

        # Perform a dummy request to verify the model is working
        test_prompt = "Hello, world!"
        response = llm.query(test_prompt)
        print("Response received from model:", response)

        print("Ollama model verification successful.")
        return True
    except Exception as e:
        print(f"Error verifying Ollama model {model_name}: {e}")
        return False

if __name__ == "__main__":
    model_name = "mistral"  # Replace with your model name
    if verify_ollama_model(model_name):
        print(f"Ollama model {model_name} is correctly installed and available.")
    else:
        print(f"Ollama model {model_name} is not correctly installed or unavailable.")
