import gradio as gr
import os

# Model configurations
MODELS = {
    "VLM": ["Qwen2-VL-7B-Instruct", "GPT-4o"],
    "Text-to-Image": ["Stable Diffusion 3", "Flux.1-Dev", "Midjourney-v6"]
}

def create_ui(process_callback):
    """
    Creates and returns the Gradio demo block.
    
    Args:
        process_callback: Async function (message, vlm_model, image_model) -> AsyncGenerator
    """
    # Custom CSS is removed as per user request, using standard Gradio components
    with gr.Blocks(title="ReasoningWithTextAndImage") as demo:
        gr.Markdown("# 🤖 Multimodal Agent Explorer")
        
        chatbot = gr.Chatbot(
            label="Conversations",
            avatar_images=(None, "https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/logo.png"),
            height=600,
        )
        
        with gr.Group():
            msg_input = gr.MultimodalTextbox(
                label="Input",
                placeholder="Type message or upload images...",
                lines=2,
                interactive=True
            )
            
            with gr.Row():
                vlm_model = gr.Dropdown(
                    choices=MODELS["VLM"],
                    label="VLM Model",
                    value=MODELS["VLM"][0]
                )
                image_model = gr.Dropdown(
                    choices=MODELS["Text-to-Image"],
                    label="Image Model",
                    value=MODELS["Text-to-Image"][0]
                )

        def user_msg(message, history):
            content = message["text"]
            for f in message["files"]:
                content += f"\n\n![User Upload]({f})" 
                
            history.append({"role": "user", "content": content})
            return history, gr.MultimodalTextbox(value=None, interactive=False)
    
        async def bot_response(history, vlm_model_name, image_model_name):
            last_msg_content = history[-1]["content"]
            dummy_msg = {"text": last_msg_content, "files": []}
            
            async for response_chunk in process_callback(dummy_msg, vlm_model_name, image_model_name):
                if history and history[-1]["role"] == "assistant":
                    history[-1] = response_chunk
                else:
                    history.append(response_chunk)
                yield history
    
        msg_input.submit(
            user_msg, [msg_input, chatbot], [chatbot, msg_input]
        ).then(
            bot_response, [chatbot, vlm_model, image_model], [chatbot]
        ).then(
            lambda: gr.MultimodalTextbox(interactive=True), None, [msg_input]
        )
        
    return demo

if __name__ == "__main__":
    # The original agent_execution function is removed and will be replaced by a process_callback
    # This __main__ block will need to be updated to provide a process_callback
    # For now, it will cause an error if run directly without a defined process_callback.
    # Example placeholder for process_callback:
    async def placeholder_process_callback(message, vlm_model_name, image_model_name):
        yield {"role": "assistant", "content": f"Received: {message['text']} with VLM: {vlm_model_name}, Image: {image_model_name}", "thought": "Placeholder thought"}

    demo = create_ui(placeholder_process_callback)
    demo.launch()
