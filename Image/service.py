import requests
import json
import os
import base64
from typing import str, Dict, List, Any
from prompt import IMAGE_PROMPT

class ImageService:
    """
    Image generation service for different models.
    """

    def __init__(self, api_client: ImageApiCall) -> None:
        self.api_client = api_client
    
    def generate_image(self, prompt: str) -> Dict[str, Any]:
        
        # Standard payloads for SD-like APIs
        payload = {
            "prompt": prompt,
            "steps": 20,
            "width": 512,
            "height": 512,
        }
        
        response = self.api_client._api_call(payload)
        
        if "images" in response:
            # Multi-image response
            return {"images": response["images"]}
        elif "image" in response:
            # Single image response
            return {"image": response["image"]}
        
        return response


class ImageApiCall:
    model_urls = {
        "Stable Diffusion 3": "http://[IP_ADDRESS]/sdapi/v1/txt2img",
        "Flux.1-Dev": "http://[IP_ADDRESS]/sdapi/v1/txt2img",
        "Midjourney-v6": "http://[IP_ADDRESS]/sdapi/v1/txt2img"
    }
    def __init__(self, model_name: str) -> None:
        self.timeout = 60 # Long timeout for generation
        self.model_name = model_name
    
    def _get_model_url(self) -> str:
        return self.model_urls.get(self.model_name)

    def _get_token(self) -> str:
        return os.getenv(self.model_name + "_API_TOKEN")

    def _api_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self._get_token()}
            response = requests.post(self._get_model_url(), headers=headers, json=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}