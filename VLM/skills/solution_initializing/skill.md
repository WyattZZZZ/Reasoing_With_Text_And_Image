---
name: solution_initializing
description: Interprets the user's intent and establishes a strategic plan for solving the problem. Use this skill ONLY at the very beginning to set the direction. Once the plan is set, move to Reasoning or Execution skills. When user's input includes "solve" or "solve the problem", you must use this skill to initialize the solution.
---

When initializing a solution, always include:

1. **Intent Interpretation & Multimodal Summary**: Restate the user's goal and provide a concise summary of all provided information, including multimodal data.
2. **Requirement Extraction**: List all constraints, given values, and target metrics.
3. **Strategic Planning**: Outline the necessary stages and tools to reach the solution using a TODO list format (e.g., `- [ ] Task`). Mark completed steps with `[x]`.
4. **Visualization Ideas (Must)**: List ideas for images to generate using the drawing tool. Format: `- Image Name: Image Prompt`.
5. **Initial Hypothesis**: State if any immediate patterns or obvious first steps are visible.
6. **Constraints (Must)**: You are only initializing the solution by planning. After generating this plan, you MUST switch to the 'Reasoning' or 'Execution' skill in the next step to begin the actual work. Do not stay in solution_initializing.

Focus on clarity and scope definition. Set the stage for the reasoning and derivation phases.
