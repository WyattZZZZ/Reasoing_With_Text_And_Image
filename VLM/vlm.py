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
import logging

logger = logging.getLogger("VLM")

@dataclass
class VlmStep:
    stage: str
    message: str
    images: List[Any] # List of PIL Images or similar
    is_final: bool = False

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
        while not self.done:
            self.round_count += 1
            last_memory = self.agent.memory.get_latest_memory()
            logger.info(f"Round {self.round_count}")
            
            if self.round_count > self.max_rounds:
                self.done = True
                msg = last_memory.get("Message", "") + "\n### Max rounds reached, You must change your skill to \"response\" to finish this question"
                yield VlmStep(stage="Response", message=msg, images=[])
                return

            # 1. Stream Skill Selection
            memory_data = None
            accumulated_skill_text = ""
            for fragment in self.agent._select_skill_and_tools_stream(last_memory):
                if isinstance(fragment, str):
                    accumulated_skill_text += fragment
                    # print(f"DEBUG: Skill Selection Fragment: {fragment}")
                    yield VlmStep(stage="Selecting Skill", message=accumulated_skill_text, images=[])
                else:
                    memory_data = fragment
            
            # Yield final state for this stage
            yield VlmStep(stage="Selecting Skill", message=accumulated_skill_text, images=memory_data.get("GeneratedImages", []) if memory_data else [], is_final=True)

            if not memory_data:
                yield VlmStep(stage="Error", message="Failed to select skill", images=[])
                return

            # 2. Stream Skill Execution
            skill_content = get_skill(memory_data["SkillSelection"])
            accumulated_run_text = ""
            for fragment in self.agent._running_stream(memory_data, skill_content):
                if isinstance(fragment, str):
                    accumulated_run_text += fragment
                    yield VlmStep(stage=memory_data["Stage"], message=accumulated_run_text, images=memory_data.get("GeneratedImages", []))
                else:
                    # Final result processed
                    pass
            
            # Yield final state for this stage
            yield VlmStep(stage=memory_data["Stage"], message=accumulated_run_text, images=memory_data.get("GeneratedImages", []), is_final=True)

            if memory_data["Stage"] == "Response":
                self.done = True
                # Yield the final step for the "Response" stage
                yield VlmStep(stage=memory_data["Stage"], message=accumulated_run_text, images=memory_data.get("GeneratedImages", []), is_final=True)


class VlmAgent:
    """
    VLM Agent core service.
    """
    def __init__(self, VLM_model: "VlmModel", image_service: Any, input: Dict[str, Any]) -> None:
        self.memory = memory.MemoryService(input)
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
    
    def run(self) -> VlmRun:
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
                    time.sleep(1)
            
            for future in as_completed(future_to_tool):
                tool_name = future_to_tool[future]
                try:
                    res = future.result()
                    results.append({"tool": tool_name, "result": res})
                except Exception as e:
                    results.append({"tool": tool_name, "error": str(e)})

        return results

    def _sanitize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure fields like Message, SkillSelection, and Stage are strings.
        If they are lists, join them with newlines.
        """
        for field in ["Message", "SkillSelection", "Stage"]:
            if field in response:
                val = response[field]
                if isinstance(val, list):
                    response[field] = "\n".join([str(item) for item in val])
                else:
                    response[field] = str(val)
        return response

    def _select_skill_and_tools_stream(self, last_memory: Dict[str, Any]) -> Iterator[str | Dict[str, Any]]:
        try:
            skill_prompt = SKILL_SELECTION_PROMPT.format(skills=get_skill_categories())
            prompt = (
                STAGE_PROMPT + "\n" + 
                "Last Memory:\n" + str(last_memory.get("Message", "")) + "\n" +
                TOOLS_PROMPT + "\n" + 
                skill_prompt + "\n" +
                RESPONSE_PROMPT
            )
            logger.info(f"DEBUG: Stage 1 Prompt:\n{prompt}")
            max_retries = 3
            response = None
            
            for attempt in range(max_retries):
                full_response = ""
                if attempt > 0:
                     msg = f"\n\n[Warning: Invalid JSON. Retrying attempt {attempt+1}/{max_retries}...]\n\n"
                     yield msg
                     
                for chunk in self.vlm_model.generate_stream(prompt, last_memory.get("Images", [])):
                    full_response += chunk
                    yield chunk
                
                # Parse final JSON
                parsed = False
                try:
                    response = json.loads(full_response)
                    parsed = True
                except:
                    # Simple heuristic if JSON is wrapped in backticks
                    if "```json" in full_response:
                        try:
                            json_str = full_response.split("```json")[1].split("```")[0].strip()
                            response = json.loads(json_str)
                            parsed = True
                        except:
                            pass
                
                if parsed:
                    break
                
                logger.warning(f"JSON Parse failed on attempt {attempt+1}")
            
            if not response:
                 # Fallback if all retries fail
                 response = {"Message": full_response, "Stage": "Thinking", "SkillSelection": "reasoning"}
            
            print(f"DEBUG: Stage 1 JSON Response:\n{json.dumps(response, indent=2, ensure_ascii=False)}")
            
            response = self._sanitize_response(response)
            self.memory.update_memory_skill_stage(response.get("SkillSelection", ""), response.get("Stage", ""))
            
            tool_results = self._tool_processing(response.get("tool_list", []))
            generated_images = []
            for res_item in tool_results:
                tool_name = res_item["tool"]
                res = res_item.get("result")
                if not res: continue
                if tool_name == "generate_image":
                    if "images" in res:
                        for img in res["images"]:
                            if img:
                                self.memory.append_image(img)
                                generated_images.append(img)
                    elif "image" in res:
                        img = res["image"]
                        if img:
                            self.memory.append_image(img)
                            generated_images.append(img)
                    elif "error" in res:
                        logger.error(f"Generate Image Error: {res['error']}")
                        print(f"DEBUG: Generate Image Error: {res['error']}")
            
            next_memory_context = self.memory.get_latest_memory()
            next_memory_context["GeneratedImages"] = generated_images
            for res_item in tool_results:
                if res_item["tool"] == "get_all_memory" and res_item.get("result"):
                    logger.info(f"DEBUG: Get All Memory Result: {res_item['result']}")
                    next_memory_context["Message"] = res_item["result"]
            
            yield next_memory_context

        except Exception as e:
            logger.error(f"Error in _select_skill_and_tools_stream: {e}")
            yield last_memory

    def _running_stream(self, last_memory: Dict[str, Any], skill_content: str) -> Iterator[str | Dict[str, Any]]:
        try:
            prompt = "SKILL INSTRUCTIONS:\n" + str(skill_content) + "\n" + str(last_memory.get("Message", ""))
            full_response = ""
            for chunk in self.vlm_model.generate_stream(prompt, last_memory.get("Images", [])):
                full_response += chunk
                yield chunk
            
            # Simple response parsing
            response = {"Message": full_response}
            response = self._sanitize_response(response)
            self.memory.append_message(response.get("Message", full_response))
            yield response
        except Exception as e:
            logger.error(f"Error in _running_stream: {e}")
            yield {"Message": f"Error: {e}"}

    def _select_skill_and_tools(self, last_memory: Dict[str, Any]) -> Dict[str, Any]:
        # Legacy/Internal method
        for res in self._select_skill_and_tools_stream(last_memory):
            if isinstance(res, dict): return res
        return last_memory

    def _running(self, last_memory: Dict[str, Any], skill_content: str) -> Dict[str, Any]:
        # Legacy/Internal method
        for res in self._running_stream(last_memory, skill_content):
            if isinstance(res, dict): return res
        return {"Message": "Error"}

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
        logger.info(f"VLM Request Start: model={self.model_name}, prompt={prompt}, images={len(images) if images else 0}")  
        return self.service.generate_text(prompt, images)

    def generate_stream(self, prompt: str, images: Optional[List[Image.Image]] = None) -> Iterator[str]:
        logger.info(f"VLM Stream Request Start: model={self.model_name}, prompt_length={len(prompt)}")
        return self.service.generate_stream(prompt, images)

        