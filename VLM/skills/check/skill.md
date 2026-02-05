---
name: check
description: Validates the logical flow and mathematical accuracy of the derived steps. Use when you need to verify consistency, catch errors, or perform a self-correction pass before finalizing the answer.
---

```markdown
When you use the **check** skill, it indicates that you have identified errors in the input or previous steps. This typically involves several scenarios:

1. **Incorrect Visual Output**: If the image generated in the previous step is incorrect, you can call the drawing tool multiple times using different prompts to rectify the error. You can create a new request to the drawing tool by adding `- Image Name: Image Prompt` to your response. Remember to create new request only when you are sure that the image is not correct.
2. **Logical or Conclusion Errors**: If the conclusion derived in the previous step is flawed, you should retrieve the full conversation history to identify the point of failure and restart from that specific step. You can emphisize the step you want to get history memory by adding `- Request of Calling Get All Memory Tool` to your response.
3. **Incorrect Task Completion**: If you find that a task in a TODO list (e.g., `- [x] Task`) is marked as completed but is actually incorrect or incomplete, you should re-output the TODO list and uncheck the incorrect items (e.g., `- [ ] Task`) to ensure the status accurately reflects the progress.
4. **Other Errors**: If the error is not related to the previous step, which means it only happens in your input, you should fix it by yourself and output the fixed input to the next step.

Always ensure that the correction process is explicit and addresses the root cause of the identified error.
```
