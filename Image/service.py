import requests
import json
import os
import base64
import io
from typing import Dict, List, Any, Optional
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath

# Attempt imports for specific providers
try:
    import dashscope
    from dashscope import ImageSynthesis
except ImportError:
    ImageSynthesis = None

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

class ImageApiCall:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Dispatch generation to the appropriate backend based on model_name.
        """
        if "Flux" in self.model_name:
            return self._generate_flux(prompt)
        else:
            # Fallback or specific mapping for HF
            # Mapping other models to HF if possible, or defaulting to User's example model
            return self._generate_hf(prompt)

    def _generate_flux(self, prompt: str) -> Dict[str, Any]:
        if not ImageSynthesis:
            return {"error": "dashscope library not installed. Install with `pip install dashscope`"}
        
        # Configure API Key
        api_key = os.getenv("FLUX_API_KEY")
        if api_key:
            dashscope.api_key = api_key
        
        try:
            rsp = ImageSynthesis.call(model="flux-schnell",
                                      prompt=prompt,
                                      size='1024*1024')
            
            if rsp.status_code == HTTPStatus.OK:
                images_b64 = []
                if hasattr(rsp.output, 'results'):
                    for result in rsp.output.results:
                        if hasattr(result, 'url'):
                            # Download content to memory
                            img_data = requests.get(result.url).content
                            b64_str = base64.b64encode(img_data).decode("utf-8")
                            images_b64.append(b64_str)
                return {"images": images_b64}
            else:
                return {"error": f"Flux Failed: {rsp.code} - {rsp.message}"}
        except Exception as e:
            return {"error": str(e)}

    def _generate_hf(self, prompt: str) -> Dict[str, Any]:
        if not InferenceClient:
            return {"error": "huggingface_hub library not installed. Install with `pip install huggingface_hub`"}
        
        token = os.getenv("HF_TOKEN")
        if not token:
             return {"error": "HF_TOKEN not set in environment."}
             
        try:
            # Using user's example provider and model
            client = InferenceClient(provider="fal-ai", api_key=token)
            # Default to "zai-org/GLM-Image" as per user example, or map others
            model = "zai-org/GLM-Image" 
            
            # Returns PIL Image
            image = client.text_to_image(prompt, model=model)
            
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {"image": img_str}
        except Exception as e:
            return {"error": str(e)}

class ImageService:
    """
    Image generation service wrapper.
    """
    def __init__(self, api_client: ImageApiCall) -> None:
        self.api_client = api_client
    
    def generate_image(self, prompt: str) -> Dict[str, Any]:
        # Delegate directly to smart client
        return self.api_client.generate(prompt)