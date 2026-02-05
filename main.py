import asyncio
import os
import sys
import time

# Ensure the path is set up to find modules
sys.path.append(os.getcwd())
from gradio.ui import create_ui
from VLM.vlm import VlmAgent, VlmModel
from VLM.service import VLMService, LocalVLMService
from Image.service import ImageService, ImageApiCall

async def agent_execution(message, vlm_model_name, image_model_name):
    """
    Executes the real VlmAgent and yields history updates.
    """
    user_text = message.get("text", "")
    user_files = message.get("files", [])
    
    print(f"Starting Agent with VLM: {vlm_model_name}, Image: {image_model_name}")
    
    # Initialize infrastructure
    vlm_model = VlmModel(vlm_model_name)
    image_api_call = ImageApiCall(image_model_name) 
    image_service = ImageService(image_api_call)
    
    # VlmAgent init: (VLM_model, image_service, question)
    agent = VlmAgent(vlm_model, image_service)
    
    # Run the agent
    for step in agent.run(user_text):
        thought_md = f"### Stage: {step.stage}\n\n{step.message}\n"
        
        # Visualize images in thinking block
        if step.images:
            thought_md += "\n#### Generated Visuals:\n"
            for i, img in enumerate(step.images):
                # Save to a temporary file for Gradio to serve
                tmp_dir = "temp_images"
                os.makedirs(tmp_dir, exist_ok=True)
                img_path = os.path.join(tmp_dir, f"gen_{int(time.time())}_{i}.png")
                img.save(img_path)
                # Use Markdown to display in thought block
                thought_md += f"![Image {i+1}](file/{img_path})\n"
        
        yield {
            "role": "assistant",
            "content": step.message if step.stage == "Response" else "Analyzing and Generating...",
            "thought": thought_md
        }

if __name__ == "__main__":
    print("Starting ReasoningWithTextAndImage Application...")
    demo = create_ui(process_callback=agent_execution)
    demo.launch()
