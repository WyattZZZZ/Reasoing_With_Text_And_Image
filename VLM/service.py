import json
import os
import base64
from typing import Dict, List, Any, Optional
import requests


import io
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def get_skill_categories() -> Dict[str, Any]:
    """
    Get skill categories for different models.
    Including skill name and skill description.
    """
    skills = {}
    if not os.path.exists("skills"):
        return skills
    for skill in os.listdir("skills"):
        skill_path = os.path.join("skills", skill, "skill.md")
        if os.path.exists(skill_path):
            with open(skill_path, "r", encoding="utf-8") as f:
                skills[skill] = f.readlines()[0:4]
    return skills

def get_skill(category: str) -> str:
    """
    Get skill for different models.
    """
    path = f"skills/{category}/skill.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Attempt imports for specific providers
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class VLMService:
    """
    VLM service for different models (API Based).
    """
    model_config = {
        "qvq-72b-preview": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "VLM_API_KEY"},
        "qwen2.5-math-1.5b-instruct": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "VLM_API_KEY"}, # User example model
        "qwen3-vl-plus": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "VLM_API_KEY"},
    }
    
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.config = self.model_config.get(model_name)
        
        # Fallback to DashScope compatible defaults if model not explicitly in map but likely Qwen
        if not self.config:
            if "qwen" in model_name.lower():
                 self.config = {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "VLM_API_KEY"}
            else:
                 # Default to OpenAI standard
                 self.config = {"base_url": "https://api.openai.com/v1", "api_key_env": "VLM_API_KEY"}

        self.api_key = os.getenv(self.config["api_key_env"])
        if not self.api_key:
             # Try fallback to specific env requested by user usage example
             self.api_key = os.getenv("DASHSCOPE_API_KEY") 

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_text(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        if not OpenAI:
            return json.dumps({"error": "openai library not installed. Install with `pip install openai`"})

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.config["base_url"]
            )
            
            messages = [{
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }]
            
            # Append images if provided
            if images:
                for img in images:
                    b64_image = self._image_to_base64(img)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                    })

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            
            return completion.choices[0].message.content

        except Exception as e:
            return json.dumps({"error": str(e)})

class LocalVLMService(VLMService):
    """
    Local VLM service running Qwen2-VL.
    """
    _model = None
    _processor = None

    def __init__(self, model_name: str) -> None: 
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        if LocalVLMService._model is None:
            # Map model names to actual paths or HF IDs
            model_id = "Qwen/Qwen2-VL-7B-Instruct" # Defaulting to Instruct
            LocalVLMService._processor = AutoProcessor.from_pretrained(model_id)
            LocalVLMService._model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto"
            )

    def generate_text(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if images:
            for img in images:
                messages[0]["content"].insert(0, {"type": "image", "image": img})

        # Preparation for inference
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        if images:
            inputs = self._processor(
                text=[text], images=images, padding=True, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self._processor(
                text=[text], padding=True, return_tensors="pt"
            ).to(self.device)

        # Generate!
        generated_ids = self._model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
