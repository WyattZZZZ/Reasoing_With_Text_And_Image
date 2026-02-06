---
name: reasoning
description: Conducts deep, step-by-step logical derivation and mathematical calculation. Use for the core problem-solving phase where you deduce new information from known facts. Only use after you find the problem is initialized, which means it is not a user input but a tool output (contains a plan or analysis).
---

```markdown
When reasoning through a problem, follow these guidelines:

1. **Planning and Progress**: If the input contains a problem-solving outline or task plan, preserve it at the beginning of your response using a TODO list format (e.g., `- [ ] Task`). Mark completed steps with `[x]`, and only finish one unfinished todo at this time.
2. **Breakdown the Complexity**: Divide the problem into smaller, logical sub-problems.
3. **Apply Analytical Rules**: Use formal logic, mathematical theorems, or scientific principles to derive steps.
4. **Chain of Thought**: Explicitly show the connection between each step.
5. **Intermediate Results**: Clearly state any values or relationships discovered along the way.
6. **Visual Reasoning (Must)**: To aid analysis, generate images by including the following format in your output: `- Image Name: Image Prompt`. You may generate multiple images to facilitate deeper investigation.
7. **Mathematical Reasoning (Must)**: To ensure accuracy, show your work clearly so each step can be easily verified, and you must show your work in a step-by-step manner.
8. **Image Analysis (Must)**: Analyze the image and provide a clear and check if the image is correct. If it is, use it for further inference, if not, indicate that the image is incorrect and provide a new command to generate a new image.

Maintain high precision. Only finish one unfinished todo at this time
```
