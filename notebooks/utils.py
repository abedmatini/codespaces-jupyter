import requests
import json
import os
from typing import List, Dict

def generate_with_single_input(prompt: str, 
                               role: str = 'user', 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               local_url: str = "http://127.0.0.1:1234",
                               local_model: str = "llama-3.2-1b-instruct",
                              **kwargs):
    
    payload = {
        "model": local_model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }
    
    # Only add temperature and top_p if they're not None
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
        
    url = f"{local_url.rstrip('/')}/v1/chat/completions"
    try:
        response = requests.post(url, json=payload)
        if not response.ok:
            raise Exception(f"Error while calling local LLM: {response.text}")
        json_dict = json.loads(response.text)
    except Exception as e:
        raise Exception(f"Failed to connect to local LLM server at {local_url}.\nException: {e}")
    
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict


def generate_with_multiple_input(messages: List[Dict], 
                               top_p: float = None, 
                               temperature: float = None,
                               max_tokens: int = 500,
                               local_url: str = "http://127.0.0.1:1234",
                               local_model: str = "llama-3.2-1b-instruct",
                               **kwargs):
    
    payload = {
        "model": local_model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    # Only add temperature and top_p if they're not None
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
        
    url = f"{local_url.rstrip('/')}/v1/chat/completions"
    try:
        response = requests.post(url, json=payload)
        if not response.ok:
            raise Exception(f"Error while calling local LLM: {response.text}")
        json_dict = json.loads(response.text)
    except Exception as e:
        raise Exception(f"Failed to connect to local LLM server at {local_url}.\nException: {e}")
    
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict
