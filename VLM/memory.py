from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Memory:
    SkillSelection: str
    Stage: str
    Message: str
    Images: List[Image.Image] # List of PIL Images

class MemoryService:
    def __init__(self, input: Dict[str, Any]) -> None:
        self.memory: List[Memory] = []
        self.input = input
        self.init_memory()

    def init_memory(self) -> None:
        self.memory.append(Memory(
            SkillSelection="", 
            Stage="Initializing",
            Message=self.input.get("text", ""),
            Images=self.input.get("files", []) # Already PIL Images from main.py
        ))

    def append_message(self, message: Any) -> None:
        if isinstance(message, list):
            message = "\n".join([str(m) for m in message])
        self.memory.append(Memory(SkillSelection="", Stage="", Message=str(message), Images=[]))

    def update_memory_skill_stage(self, skill: str, stage: str) -> None:
        """
        Update memory with new skill and stage.
        """
        if self.memory:
            self.memory[-1].SkillSelection = skill
            self.memory[-1].Stage = stage

    def append_image(self, image: Image.Image) -> None:
        """
        Add a new PIL image to the current memory entry.
        """
        if self.memory:
            if image:
                self.memory[-1].Images.append(image)

    def update_message(self, message: Any) -> None:
        """
        Update memory with new message.
        """
        if self.memory:
            if isinstance(message, list):
                message = "\n".join([str(m) for m in message])
            self.memory[-1].Message = str(message)

    def get_latest_memory(self) -> Dict[str, Any]:
        if not self.memory:
            return {}
        data = self.memory[-1] 
        return {
            "SkillSelection": data.SkillSelection,
            "Stage": data.Stage,
            "Message": data.Message,
            "Images": data.Images
        }

    def get_all_memory(self) -> Dict[str, Any]:
        """
        Produce a summary of all memory entries.
        """
        if not self.memory:
            return {}
        
        # Concatenate all messages with index
        all_messages = "\n".join([f"No.{i}: {m.Message}" for i, m in enumerate(self.memory)])
        # Collect all images across all memory entries
        all_images = []
        for m in self.memory:
            for img in m.Images:
                if img:
                    all_images.append(img)
        
        return {
            "SkillSelection": self.memory[-1].SkillSelection,
            "Stage": self.memory[-1].Stage,
            "Message": all_messages,
            "Images": all_images
        }
