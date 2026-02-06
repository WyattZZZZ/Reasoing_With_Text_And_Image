---
name: response
description: Formulates the final answer for the user. Use when the problem is solved and you are ready to present the conclusion clearly and concisely.
---

```markdown
When providing a response, always include:

1. **Direct Answer**: Start with the clear, final conclusion or value.
2. **Summary of Logic**: Briefly recap the key reasoning steps that led to the answer.
3. **Format Compliance**: Ensure the output matches the requested format (e.g., Markdown, specific units, etc.).
4. **Visual Integration (MUST)**: If tools generated images, reference them clearly in the final text.

(Must) This is the final step of the entire analysis process. You must use `- Request of Calling Get All Memory Tool` to retrieve all previous conversation history to create a summary.

(Must) You can also create images to help you summarize better. The method for creating images is:
`- Image Name: Image Prompt`

Keep responses helpful and professional. Avoid repeating the entire thinking process, focus on the result.
```
