import gradio as gr
import time

# Model configurations
MODELS = {
    "VLM": ["Qwen-2-VL-7B", "GPT-4o", "Claude-3.5-Sonnet"],
    "Text-to-Image": ["Stable Diffusion 3", "Flux.1-Dev", "Midjourney-v6"]
}

def mock_agent_response(message, category, model_name):
    """
    Simulates an agent's response with a thinking process.
    """
    user_text = message.get("text", "")
    user_files = message.get("files", [])
    
    # 1. Yield an initial thinking state
    thought_process = f"Using {model_name} ({category})...\nStarting analysis...\n"
    if user_files:
        thought_process += f"Found {len(user_files)} image(s). Processing visual features...\n"
    
    yield [
        {"role": "assistant", "content": "", "thought": thought_process}
    ]
    time.sleep(1)
    
    # 2. Add intermediate steps
    thinking_image_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/guides/assets/logo.png"
    thought_process += f"![Reasoning Step]({thinking_image_url})\n"
    thought_process += f"Thinking step: {model_name} is evaluating the prompt: '{user_text}'\n"
    
    yield [
        {"role": "assistant", "content": "Analyzing...", "thought": thought_process}
    ]
    time.sleep(1)
    
    # 3. Final Output based on category
    if category == "VLM":
        final_content = f"### {model_name} Response\n\nYou asked: '{user_text}'\n"
        if user_files:
            final_content += f"I see {len(user_files)} image(s) and I can help you understand them."
        else:
            final_content += "Please upload an image for more specific VLM analysis."
    else:
        final_content = f"### {model_name} Generation\n\nPrompt: '{user_text}'\n"
        gen_image_url = "https://picsum.photos/400/300"
        final_content += f"Here is the generated image:\n\n![Generated Image]({gen_image_url})"
        
    yield [
        {"role": "assistant", "content": final_content, "thought": thought_process}
    ]

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
            model_category = gr.Dropdown(
                choices=list(MODELS.keys()),
                label="Model Category",
                value=list(MODELS.keys())[0]
            )
            model_specific = gr.Dropdown(
                choices=MODELS[list(MODELS.keys())[0]],
                label="Specific Model",
                value=MODELS[list(MODELS.keys())[0]][0]
            )

    def update_models(category):
        return gr.Dropdown(choices=MODELS[category], value=MODELS[category][0])

    model_category.change(update_models, inputs=[model_category], outputs=[model_specific])
    
    def user_msg(message, history):
        content = message["text"]
        for f in message["files"]:
            # Displaying image directly in chatbot using markdown or component
            content += f"\n\n![User Upload]({f})" 
            
        history.append({"role": "user", "content": content})
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def bot_response(history, category, model_name):
        # We need the original multimodal input to pass to the mock agent
        # For simplicity in this demo, we'll just use the last history entry text
        # In a real app, you'd handle file paths properly
        last_msg = history[-1]["content"]
        
        dummy_msg = {"text": last_msg, "files": []} # Mocked files for logic
        
        for response_chunk in mock_agent_response(dummy_msg, category, model_name):
            if history and history[-1]["role"] == "assistant":
                history[-1] = response_chunk[0]
            else:
                history.append(response_chunk[0])
            yield history

    msg_input.submit(
        user_msg, [msg_input, chatbot], [chatbot, msg_input]
    ).then(
        bot_response, [chatbot, model_category, model_specific], [chatbot]
    ).then(
        lambda: gr.MultimodalTextbox(interactive=True), None, [msg_input]
    )

if __name__ == "__main__":
    demo.launch()
