import base64
import io
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Memory:
    SkillSelection: str
    Stage: str
    Message: str
    Images: List[str] # List of Base64 strings

class MemoryService:
    def __init__(self, question: str) -> None:
        self.memory: List[Memory] = []
        self.question = question
        self.init_memory()

    def init_memory(self) -> None:
        self.memory.append(Memory(SkillSelection="", Stage="Initializing", Message=self.question, Images=[]))

    def append_message(self, message: str) -> None:
        self.memory.append(Memory(SkillSelection="", Stage="", Message=message, Images=[]))

    def update_memory_skill_stage(self, skill: str, stage: str) -> None:
        """
        Update memory with new skill and stage.
        """
        if self.memory:
            self.memory[-1].SkillSelection = skill
            self.memory[-1].Stage = stage

    def append_image(self, image_bytes: bytes) -> None:
        """
        Add a new image to the current memory entry.
        """
        if self.memory:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else ""
            if img_b64:
                self.memory[-1].Images.append(img_b64)

    def update_message(self, message: str) -> None:
        """
        Update memory with new message.
        """
        if self.memory:
            self.memory[-1].Message = message

    def _b64_to_pil(self, b64_str: str) -> Any:
        if not b64_str:
            return None
        # Base64 to PIL Image
        img_data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_data))

    def get_latest_memory(self) -> Dict[str, Any]:
        if not self.memory:
            return {}
        data = self.memory[-1] 
        return {
            "SkillSelection": data.SkillSelection,
            "Stage": data.Stage,
            "Message": data.Message,
            "Images": [self._b64_to_pil(img) for img in data.Images]
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
            for img_b64 in m.Images:
                pil_img = self._b64_to_pil(img_b64)
                if pil_img:
                    all_images.append(pil_img)
        
        return {
            "SkillSelection": self.memory[-1].SkillSelection,
            "Stage": self.memory[-1].Stage,
            "Message": all_messages,
            "Images": all_images
        }
