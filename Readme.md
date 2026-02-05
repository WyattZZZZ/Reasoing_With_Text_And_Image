# Reasoning With Text and Image

An advanced Multimodal Reasoning Agent explorer featuring iterative thinking, self-correction, and parallel tool execution. This project allows users to interact with state-of-the-art VLMs (Vision-Language Models) and T2I (Text-to-Image) generators through a unified interface.

## 🚀 Quick Start

### Using uv (Recommended)
1. **Setup & Run**:
   ```bash
   uv run main.py
   ```
   *`uv` will automatically handle environment creation and dependency installation.*

### Using pip
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Application**:
   ```bash
   python main.py
   ```

## ⚙️ Configuration
Create a `.ENV` file based on `.ENV.example` and add your API keys:

## 🧠 Key Features

- **Iterative Reasoning**: Uses a "Think-Act-Step" loop with specific skills like `reasoning`, `check`, and `solution_initializing`.
- **Multimodal Feedback**: The agent can generate images to aid its own visual reasoning or verify its output.
- **Parallel Tool Execution**: Uses a synchronous threaded architecture (`ThreadPoolExecutor`) to run multiple tools (image generation, memory retrieval) simultaneously.
- **Memory Management**: Structured conversation history that tracks stages, messages, and multiple image objects.
- **Model Flexibility**: Supports both remote API models and local inference fallbacks.

## 🛠️ Architecture

The project follows a modular design separating the UI, Agent logic, and Service providers:

- **Frontend**: Built with **Gradio**, providing a real-time "Thinking" visualization and chatbot experience.
- **Agent Core (`VLM/`)**: Manages the reasoning state machine and skill selection.
- **Service Layer**: 
  - **VLM Service**: Uses the OpenAI SDK to interface with DashScope and OpenAI models.
  - **Image Service**: Direct integration with DashScope (Flux) and HuggingFace Hub (InferenceClient).

## 📁 Project Structure

```text
├── main.py              # Application entry point & orchestration
├── gradio/
│   └── ui.py            # UI layout and component logic
├── VLM/
│   ├── vlm.py           # Agent core logic (Run & Agent classes)
│   ├── service.py       # VLM API & Local model integrations
│   ├── memory.py        # Conversation and visual memory service
│   └── skills/          # Markdown-defined agent skills
├── Image/
│   └── service.py       # Flux & HuggingFace generation engines
└── prompt.py            # System prompts and tool definitions
```

## 🤖 Supported Models

### Multimodal Reasoning (VLM)
- **qvq-72b-preview**: High-performance reasoning model.
- **qwen2.5-math-1.5b-instruct**: Remote reasoning model.
- **qwen3-vl-plus**: Remote vision-language reasoning model.
- **qwen2-VL-7B-Instruct**: Local vision-language reasoning model.

### Image Generation (T2I)
- **Flux (DashScope)**: High-quality image synthesis.
- **HuggingFace-fal**: Access to GLM-Image and other HF-hosted models.

## 📄 License
MIT