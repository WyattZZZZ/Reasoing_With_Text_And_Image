import requests
import json
import os
import base64
import io
from PIL import Image
from typing import Dict, List, Any, Optional
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import logging

logger = logging.getLogger("ImageService")

# Attempt imports for specific providers
try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None

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
        logger.info(f"Image Request: model={self.model_name}, prompt_length={len(prompt)}")
        
        # DashScope supported models
        dashscope_models = ["stable-diffusion-3.5-large-turbo", "qwen-image-max"]
        
        if any(m in self.model_name for m in dashscope_models):
            return self._generate_dashscope(prompt)
        else:
            # Fallback or specific mapping for HF
            # Mapping other models to HF if possible, or defaulting to User's example model
            return self._generate_hf(prompt)

    def _generate_dashscope(self, prompt: str) -> Dict[str, Any]:
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key
            
        model = "qwen-image-max" 
        size = "1328*1328"   
        
        if "stable-diffusion-3.5-large-turbo" in self.model_name:
            model = "stable-diffusion-3.5-large-turbo"
            size = "1024*1024"
        elif "qwen-image-max" in self.model_name:
            model = "qwen-image-max"
            size = "1328*1328" # Specific size requested by user
            
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ]
            response = MultiModalConversation.call(
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        result_format='message',
                        stream=False,
                        watermark=False,
                        prompt_extend=True,
                        negative_prompt="Low resolution, low image quality, blurry images, unclear mathematical symbols, unclear mathematical formulas, incomplete mathematical symbols, incorrect mathematical symbols, unclear mathematical diagrams, excessive smoothing, the image has an artificial intelligence feel. Chaotic composition. Text is blurry and distorted.",
                        size=size
                        )
            
            if response.status_code == HTTPStatus.OK:
                logger.info(f"DashScope Response: {response}")
                url = response.output.choices[0].message.content[0]["image"]
                img_data = requests.get(url).content
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images = [img]
                logger.info(f"DashScope Response: success, model={model}, num_images={len(images)}")
                return {"images": images}
            else:
                error_msg = f"DashScope Failed: {response.code} - {response.message}"
                logger.error(error_msg)
                return {"error": error_msg}
        except Exception as e:
            logger.exception("DashScope Error")
            return {"error": str(e)}

    def _generate_hf(self, prompt: str) -> Dict[str, Any]:
        
        token = os.getenv("HF_TOKEN")
        if not token:
             return {"error": "HF_TOKEN not set in environment."}
             
        try:
            # Using user's example provider and model
            client = InferenceClient(provider="fal-ai", api_key=token)
            # Default to "zai-org/GLM-Image" as per user example, or map others
            model = "zai-org/GLM-Image" 
            
            # Returns PIL Image
            image = client.text_to_image(prompt, model=model).convert("RGB")
            images = [image]
            logger.info("HF Image Response: success")
            return {"images": images}
        except Exception as e:
            logger.exception("HF Image Error")
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