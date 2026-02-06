import asyncio
import os
import sys
import time
import logging
from dotenv import load_dotenv
import gradio as gr
from PIL import Image
from typing import Dict, List, Any, Optional

# --- Logging & Imports (unchanged) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ReasoningApp")
sys.path.append(os.getcwd())

from VLM.vlm import VlmAgent, VlmModel
from Image.service import ImageService, ImageApiCall

# --- The "Big Message" Logic ---

def agent_execution(message: Dict[str, Any], history: List[Any], vlm_model_name: str, image_model_name: str):
    user_input = {
        "text": message.get("text", ""),
        "files": [Image.open(f["path"] if isinstance(f, dict) else f).convert("RGB") for f in message.get("files", [])]
    }
    vlm_model = VlmModel(vlm_model_name)
    image_api_call = ImageApiCall(image_model_name) 
    image_service = ImageService(image_api_call)
    agent = VlmAgent(vlm_model, image_service, user_input)
    
    finalized_blocks = [] 
    current_thought = ""

    for step in agent.run():
        if step.stage == "Selecting Skill":
            current_thought = step.message
        
        display_content = list(finalized_blocks) 
        
        current_status = f"**[{step.stage}]**\n{step.message}"
        
        if display_content and isinstance(display_content[-1], str):
            display_content[-1] += f"\n\n{current_status}"
        else:
            display_content.append(current_status)

        if step.is_final:
            finalized_blocks.append(f"✅ **{step.stage} Finished**\n{step.message}\n")
            
            if step.images:
                for i, img in enumerate(step.images):
                    tmp_dir = os.path.abspath("temp_images")
                    os.makedirs(tmp_dir, exist_ok=True)
                    img_path = os.path.join(tmp_dir, f"gen_{int(time.time()*1000)}_{i}.png")
                    img.save(img_path)
                    
                    finalized_blocks.append({"path": img_path, "alt": f"Result {i}"})
                
                finalized_blocks.append("\n---\n")
            
            display_content = list(finalized_blocks)

        yield {
            "role": "assistant",
            "content": display_content,
            "thought": current_thought
        }

    yield {
        "role": "assistant",
        "content": finalized_blocks,
        "thought": current_thought
    }

# --- UI Definition ---

MODELS = {
    "VLM": ["qwen3-vl-plus", "qwen2.5-math-1.5b-instruct", "qvq-72b-preview", "Qwen2-VL-7B-Instruct"],
    "Text-to-Image": ["qwen-image-max", "stable-diffusion-3.5-large-turbo", "HuggingFace-fal"]
}

def create_ui(process_callback):
    with gr.Blocks(title="Multimodal Agent") as demo:
        gr.Markdown("# 🤖 Multimodal Agent Explorer")
        
        chatbot = gr.Chatbot(
            label="VLM Agent",
            height=800,
            render_markdown=True,
            latex_delimiters=[
                {"left": "$", "right": "$", "display": True},
                {"left": "$$", "right": "$$", "display": True},
                {"left": "\\(", "right": "\\)", "display": True},
                {"left": "\\[", "right": "\\]", "display": True},
            ]
        )
        
        gr.ChatInterface(
            fn=process_callback,
            chatbot=chatbot,
            multimodal=True,
            additional_inputs=[
                gr.Dropdown(choices=MODELS["VLM"], label="VLM Model", value=MODELS["VLM"][0]),
                gr.Dropdown(choices=MODELS["Text-to-Image"], label="Image Model", value=MODELS["Text-to-Image"][0])
            ]
        )
        
    return demo

if __name__ == "__main__":
    load_dotenv()
    demo = create_ui(agent_execution)
    demo.launch()