import os
import base64
import json
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Iterator, Dict, List, Any
from PIL import Image
from . import memory
from .service import get_skill_categories, get_skill, VLMService, LocalVLMService
from prompt import SKILL_SELECTION_PROMPT, RESPONSE_PROMPT, STAGE_PROMPT, TOOLS_PROMPT

@dataclass
class VlmStep:
    stage: str
    message: str
    images: List[Image.Image] # List of PIL Images

class VlmRun:
    """
    Iterator class to execute agent stages and return results.
    Allows for multiple rounds and self-correction.
    """
    def __init__(self, agent: "VlmAgent"):
        self.agent = agent
        self.round_count = 0
        self.max_rounds = 10
        self.done = False

    def __iter__(self) -> Iterator[VlmStep]:
        return self

    def __next__(self) -> VlmStep:
        if self.done:
            # We already returned the done step or are finished
            raise StopIteration
            
        self.round_count += 1
        last_memory = self.agent.memory.get_latest_memory()
        
        if self.round_count > self.max_rounds:
            self.done = True
            last_memory["Message"] += "\n### Max rounds reached, You must change your skill to \"response\" to finish this question"
            self.agent._select_skill_and_tools(last_memory)
            return VlmStep(stage="done", message="Max rounds reached", images=[])
        else:
            # Select skill and tools
            memory_data = self.agent._select_skill_and_tools(last_memory)
            
            # Execute the skill (Thinking or Response)
            skill_content = get_skill(memory_data["SkillSelection"])
            result = self.agent._running(memory_data, skill_content)
            
            if memory_data["Stage"] == "Response":
                self.done = True
                
            return VlmStep(
                stage=memory_data["Stage"], 
                message=memory_data["Message"], 
                images=memory_data["Images"]
            )

class VlmAgent:
    """
    VLM Agent core service.
    """
    def __init__(self, VLM_model: "VlmModel", image_service: Any) -> None:
        self.memory = None
        self.vlm_model = VLM_model
        self.image_service = image_service
        self.TOOLS = {
            "memory": {
                "get_all_memory": {"function": self.memory.get_all_memory, "params": {}}
            },
            "image_service": {
                "generate_image": {"function": self.image_service.generate_image, "params": {"prompt": ""}}
            }
        }
    
    def run(self, prompt: str) -> VlmRun:
        self.memory = memory.MemoryService(prompt)
        return VlmRun(self)

    def _tool_processing(self, tool_list: List[Dict]) -> List[Dict[str, Any]]:
        if not tool_list:
            return []
        
        exceptions = []
        results = []
        
        with ThreadPoolExecutor() as executor:
            future_to_tool = {}
            for tool_item in tool_list:
                category = tool_item.get("category")
                name = tool_item.get("name")
                
                if category in self.TOOLS and name in self.TOOLS[category]:
                    tool_info = self.TOOLS[category][name]
                    func = tool_info["function"]
                    base_params = tool_info["params"].copy()
                    
                    # Merge dynamic params
                    if tool_item.get("params"):
                        for k, v in tool_item["params"].items():
                            if k in base_params:
                                base_params[k] = v
                    
                    # Store tool name with future
                    future = executor.submit(func, **base_params)
                    future_to_tool[future] = name
            
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    res = future.result()
                    results.append({"tool": tool_name, "result": res})
                except Exception as e:
                    results.append({"tool": tool_name, "error": str(e)})

        return results

    def _select_skill_and_tools(self, last_memory: Dict[str, Any]) -> Dict[str, Any]:
        try:
            skill_prompt = SKILL_SELECTION_PROMPT.format(skills=get_skill_categories())
            prompt = (
                STAGE_PROMPT + "\n" + 
                "CURRENT CONTEXT:\n" + last_memory["Message"] + "\n" +
                TOOLS_PROMPT + "\n" + 
                skill_prompt
            )
            
            response_raw = self.vlm_model.generate_text(prompt, last_memory["Images"])
            # Assuming VLM returns JSON
            response = response_raw if isinstance(response_raw, dict) else json.loads(response_raw)
            
            self.memory.update_memory_skill_stage(response.get("SkillSelection", ""), response.get("Stage", ""))
            
            # Process tools in parallel (Synchronous but threaded)
            tool_results = self._tool_processing(response.get("tool_list", []))
            
            # Update memory based on tool results
            for res_item in tool_results:
                tool_name = res_item["tool"]
                res = res_item.get("result")
                if not res: continue
                
                if tool_name == "generate_image":
                    # Handle multi-image format from ImageService
                    if "images" in res:
                        for img_b64 in res["images"]:
                            img_bytes = base64.b64decode(img_b64)
                            self.memory.append_image(img_bytes)
                    elif "image" in res:
                        img_bytes = base64.b64decode(res["image"])
                        self.memory.append_image(img_bytes)
            
            # Default memory is just the latest state
            next_memory_context = self.memory.get_latest_memory()
            
            # Check if tools returned any data that should override the context (e.g. get_all_memory)
            for res_item in tool_results:
                tool_name = res_item["tool"]
                res = res_item.get("result")
                if not res: continue
                
                if tool_name == "get_all_memory":
                    print("DEBUG: Injecting full memory history into context.")
                    next_memory_context = res
            
            return next_memory_context

        except Exception as e:
            print(f"Error in _select_skill_and_tools: {e}")
            return last_memory

    def _running(self, last_memory: Dict[str, Any], skill_content: str) -> Dict[str, Any]:
        try:
            prompt = "SKILL INSTRUCTIONS:\n" + skill_content + "\n" + last_memory["Message"]
            response_raw = self.vlm_model.generate_text(prompt, last_memory["Images"])
            
            response = response_raw if isinstance(response_raw, dict) else {"Message": response_raw}
            # Handle list-based output or string
            msg = response.get("Message", str(response))
            self.memory.append_message(msg)
            return response
        except Exception as e:
            print(f"Error in _running: {e}")
            return {"Message": f"Error: {e}"}

class VlmModel:
    Model_service = {
        "qwen2-vl": LocalVLMService,
        "gpt-4o": VLMService,
    }
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # Find which service class to use
        service_class = None
        for key, cls in self.Model_service.items():
            if key in model_name.lower():
                service_class = cls
                break
        if not service_class:
            service_class = VLMService
            
        self.service = service_class(model_name)

    def generate_text(self, prompt: str, images: Optional[List[Image.Image]] = None) -> Any:
        return self.service.generate_text(prompt, images)





    
        