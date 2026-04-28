import os
import sys
import traceback
from openai import AzureOpenAI

# Ensure 'src' is in sys.path for absolute imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.logger_config import logger
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT

def azure_openai_client():
    """
    Returns an instance of the AzureOpenAI client.
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    return client

def chat_completion(messages, max_tokens=2000, temperature=1e-9, task_type="qna_extraction"):
    """
    Sends a chat completion request to the Azure OpenAI LLM.
    
    :param messages: List of messages in the chat format
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Sampling temperature for randomness
    :return: The content of the response message
    """
    client = azure_openai_client()
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        if hasattr(response, "choices"):
            llm_response = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            logger.info(f"{task_type} extraction completed successfully.")
        else:
            logger.error(f"{task_type}: Unexpected response type from OpenAI: {type(response)} - {response}")
            llm_response = None
        return llm_response
    except Exception as e:
        logger.error(f"{task_type}: [ERROR] Failed to get chat completion: {e}")
        traceback.print_exc()
        return None

def sample_llm_response():
    """
    Sends a sample prompt to the Azure OpenAI LLM and prints the response.
    Handles errors for better readability.
    """
    prompt = "Hello, how are you?"
    try:
        messages=[{"role": "user", "content": prompt}]  
        llm_response = chat_completion(messages=messages)
        if llm_response:
            print(f"LLM Response: {llm_response}")
        else:
            print("No response received from the LLM.")
    except Exception as e:
         print(f"[ERROR] Failed to get LLM response: {e}")
         traceback.print_exc()

if __name__ == "__main__":
    sample_llm_response()