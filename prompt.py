SKILL_SELECTION_PROMPT = """
    Here are the available skills: {skills}
    Please select the skill that best suits the last memory.
"""

TOOLS_PROMPT = """
    Here are the available tools: 
    {
        "memory": {
            "get_all_memory": {"function": "get_all_memory", "params": {}}
        },
        "image_service": {
            "generate_image": {"function": "generate_image", "params": {"prompt": "Your Prompt"}}
        }
    }
    Please select the tool that best suits the last memory.
    Tool usage:
    1. For each tool, you should know the params of the tool.
    2. Follow the tool_list in the response format to use the tool.
    3. The first level of the tool_list is category, the second level is name, the third level are function and params.
    4. For generate_image tool, you can use it as many times as you want via creating a new tool_list for each time, but you need to create different prompt for each request based on the last memory.
"""

STAGE_PROMPT = """
    Here are four stages: 
    1. Thinking: If you are discovering the solution, you should use this stage to think and plan your next step.
    2. Response: If you have found the solution, you should use this stage to respond to the user.
"""

RESPONSE_PROMPT = """
    You are an math assistant.
    What you need to do is to respond to the user based on the last memory.
    You must generate a tool_list to use the tool by review input's visualization ideas.
    For Message, It's just a simple version of the last memory's message that you read.
    Here is the response format:
    {
        "SkillSelection": "skill_name",
        "Stage": "stage_name",
        "Message": "message",
        "tool_list": [
            {
                "category": "category",
                "name": "name",
                "params": {}
            }
        ]
    }
"""

IMAGE_PROMPT = """
    You are an math image generation assistant.
    What you need to do is to generate an image based on the math problem description below:
    
"""