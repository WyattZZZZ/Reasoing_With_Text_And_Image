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

class VLMService:
    """
    VLM service for different models (API Based).
    """
    model_url = {
        "Qwen-VL-Max": "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        "GPT-4o": "https://api.openai.com/v1/chat/completions",
    }
    
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.api_key = os.getenv("VLM_API_KEY")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_text(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        url = self.model_url.get(self.model_name)
        if not url:
            return json.dumps({"error": "Model URL not found"})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Handle different API formats (simplified OpenAI-like vs DashScope)
        if "openai" in url or "GPT" in self.model_name:
            payload = {
                "model": self.model_name.lower(),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }]
            }
            if images:
                for img in images:
                    b64_image = self._image_to_base64(img)
                    payload["messages"][0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                    })
        else:
            # DashScope format as fallback/default for Qwen-VL-Max
            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ]
                }
            }
            if images:
                for img in images:
                    b64_image = self._image_to_base64(img)
                    payload["input"]["messages"][0]["content"].append({"image": f"data:image/png;base64,{b64_image}"})

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            res_data = response.json()
            # Extract content based on API
            if "choices" in res_data:
                return res_data["choices"][0]["message"]["content"]
            elif "output" in res_data:
                return res_data["output"]["choices"][0]["message"]["content"][0]["text"]
            return json.dumps(res_data)
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
