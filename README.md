# Synthetic Persona Bank (SPB)

This repository contains the generation code and information for the **Synthetic Persona Bank (SPB)** dataset.

SPB is a collection of 5,000 synthetically generated, fictional character personas. Each persona is provided in a structured JSON format and includes a name, age, personality traits, a concise background story, and a described chatting style, making them ideal for online conversational applications.

The dataset was created using a multi-stage programmatic pipeline driven by the [Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) large language model. This process was designed to produce a diverse and high-quality set of characters suitable for a wide range of tasks.

This dataset is the foundational character source for the [Synthetic Online Conversations (SOC-2508)](https://huggingface.co/datasets/marcodsn/SOC-2508) dataset and is ideal for training and evaluating language models on tasks requiring:
- Persona consistency and adoption
- Character-driven role-playing
- Stylistic and long-form dialogue generation

## Key Features

-   **Structured & Detailed:** Each persona is a rich JSON object with fields for name, age, traits, background, and a specific chatting style.
-   **Conversation-Focused:** Personas are designed with online interactions in mind, featuring explicit `chatting_style` descriptions to guide dialogue generation.
-   **Realistic Distribution:** Profession assignments are weighted based on U.S. Bureau of Labor Statistics (BLS) data to ensure a realistic demographic spread.
-   **Novelty & Diversity:** The generation process uses modified iterative sampling with few-shot examples to prevent repetition and encourage unique character creation across the 5,000 entries.
-   **Programmatic Generation:** A fully scripted pipeline ensures scalability and consistency, with all source code and seed data available.

## Getting Started

You can easily load and use this dataset with the 🤗 `datasets` library.

```bash
pip install datasets
```

```python
from datasets import load_dataset

# Load the dataset from the Hugging Face Hub
dataset = load_dataset("marcodsn/SPB-2508")

# The dataset has a single 'train' split
train_dataset = dataset['train']

# Print the first persona
print(train_dataset[0])
```

## Dataset Structure

The dataset is a single JSONL file (`data.jsonl`), where each line is a JSON object representing a single persona.

### Data Schema

Each JSON object has the following structure:

```json
{
  "name": "Elias Vance",
  "username": "quantum_scribe",
  "age": 42,
  "traits": [
    "analytical",
    "introspective",
    "witty",
    "reserved"
  ],
  "background": "A theoretical physicist who, after a breakthrough, left academia to write science fiction novels from a secluded cabin. He's currently grappling with a severe case of writer's block for his second book.",
  "chatting_style": "Uses precise language and often employs metaphors from physics. Tends to write in well-structured, complete sentences, even in casual chat.",
  "model": "Qwen3-235B-A22B-Instruct-2507",
  "id": "4436437d368e4325a7c1c6f7092c2d9e"
}
```

### Field Descriptions

-   `name` (string): The full name of the persona.
-   `username` (string, nullable): A potential online username for the persona, used to prevent the model from generating usernames within the `name` field.
-   `age` (int): The age of the persona, often constrained by the assigned profession for realism.
-   `traits` (list of strings): A list of 3-5 adjectives describing the character's core personality.
-   `background` (string): A short (1-2 sentence) background story integrating the persona's profession, life context, and age.
-   `chatting_style` (string): A brief description of the persona's typical online communication style.
-   `model` (string): The model used to generate the persona.
-   `id` (string): A unique identifier (UUID) for the persona.

## Generation Process

The personas were generated using a programmatic pipeline inspired by techniques from the [ConvoGen](https://huggingface.co/papers/2503.17460) paper:

1.  **Stage 1: Component Seeding**
    -   The process starts with a `persona_components.json` file containing weighted lists of `professions`, `life_contexts`, `traits`, and `chatting_quirks`.
    -   > Profession weights are adjusted to U.S. Bureau of Labor Statistics (BLS) data. Some professions also specify an age range to prevent unrealistic combinations (e.g., a 19-year-old retiree).

2.  **Stage 2: Dynamic Prompting & Generation**
    -   For each persona, a unique prompt is constructed by randomly selecting components (e.g., a profession, a life context, several traits).
    -   This prompt is sent to the LLM to generate the structured persona data.

3.  **Stage 3: Ensuring Novelty**
    -   To avoid generating repetitive content, the prompt includes several recently generated personas as few-shot examples, instructing the model to create something different.
    -   A basic similarity check is performed on the newly generated persona against its references to discard simple copies or highly similar concepts.
    -   The pool of few-shot examples is periodically re-seeded from a high-quality initial list to prevent stylistic drift.

4.  **Stage 4: Collection & Finalization**
    -   Valid and unique personas are added to the final pool, which is saved as a single `data.jsonl` file.

The generation scripts and seed data can be found in this repository: [github.com/marcodsn/SPB/tree/2508](https://github.com/marcodsn/SPB/tree/2508).

## Known Limitations

-   **Inherited Bias**: As a synthetic dataset, it may contain and amplify biases or stereotypes present in the base LLM used for generation.
-   **Narrative Depth**: The background and chatting style descriptions are intentionally brief. They provide a starting point but lack the depth of fully developed character biographies.
-   **Generation Patterns**: Despite efforts to ensure novelty, the generation process may fall into subtle patterns or tropes over 5,000 iterations.

## Licensing

-   The **dataset** is released under the [**Creative Commons Attribution 4.0 International (CC BY 4.0)**](https://creativecommons.org/licenses/by/4.0/) license.
-   The **generation code** in this repository is released under the [**Apache 2.0 License**](LICENSE).

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@misc{marcodsn_2025_SPB2508,
  title     = {Synthetic Persona Bank},
  author    = {Marco De Santis},
  year      = {2025},
  month     = {August},
  url       = {https://huggingface.co/datasets/marcodsn/SPB-2508},
}
```
