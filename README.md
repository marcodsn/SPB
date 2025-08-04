# Dataset Card for Synthetic Persona Bank

## Dataset Summary

This dataset contains around 5.000 synthetically generated, fictional character personas in a structured JSON format, with a focus on online-based conversational personas. Each persona includes a name, age, personality traits, a concise background story, and a described chatting style. An entry is added to idetify the model used to generate the persona.

The dataset was created programmatically using a large language model (specifically, for this iteration, we used [`Kimi-K2-Instruct`](https://huggingface.co/moonshotai/Kimi-K2-Instruct) and [`GLM-4.5-Air-FP8`](https://huggingface.co/zai-org/GLM-4.5-Air-FP8)) guided by a detailed, component-based prompting strategy.

This dataset is designed for infering language models in tasks requiring character consistency, role-playing, and stylistic dialogue generation. It is the foundation of a coming dataset containing synthetic conversations between these personas.

## Dataset Structure

The dataset consists of a single JSON file containing a list of persona objects.

### Data Instances

Each line in the dataset is a JSON object representing a single persona. Here is an example of what a persona object looks like:

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
  "chatting_style": "Uses precise language and often employs metaphors from physics. Tends to write in well-structured, complete sentences, even in casual chat."
}
```

### Data Fields

The JSON objects contain the following fields:

- **name** (string): The full name of the persona. Generated from lists of common first and last names.
- **username** (string, nullable): A potential online username for the persona. Generated from a seed list. Can be null. This was added mainly to avoid the model generating usernames inside the persona's name (which we found was very common in our tests).
- **age** (int): The age of the persona, between 19 and 75.
- **traits** (list[string]): A list of 3-5 adjectives that describe the core personality of the character.
- **background** (string): A short (1-2 sentence, ≤300 characters) background story that integrates the persona's profession, life context, and age into a coherent narrative.
- **chatting_style** (string): A brief description (≤120 characters) of the persona's typical texting or online communication style.

### Data Splits

The dataset is provided as a single file, data.json, which constitutes the train split. Users are encouraged to create their own validation and test splits as needed for their specific use case.

## Dataset Creation

### Curation Rationale

The primary motivation for creating this dataset was to generate a large-scale, diverse, and structured collection of fictional characters. Such data is invaluable for developing conversational AI that can adopt and maintain a consistent persona over long interactions, and to create derived datasets like natural conversation datasets.

### Source Data

This is a synthetically generated dataset. It was not derived from any pre-existing corpus of human-written text, but was created through a programmatic generation pipeline.

#### Generation Process

The personas were generated using the following pipeline:

1. **Component Seeding**: The process starts with a persona_components.json file containing weighted lists of professions, life_contexts, traits, and chatting_quirks.
2. **Iterative Generation**: The script iteratively generates new personas in a loop until it reaches the target number.
3. **Dynamic Prompting**: For each new persona, a unique prompt is constructed by randomly selecting components (e.g., a profession, a life context, several traits).
4. **Modified Iterative Sampling**: To avoid generating repetitive content, the prompt includes different recently generated personas as few shots examples at each iteration, as seen in the recent [ConvoGen paper](https://huggingface.co/papers/2503.17460), used to instruct the model to create something different. Additionally (the "novelty"), the script periodically re-seeds its examples from a high-quality initial list to prevent drift.
5. **LLM Generation**: The prompt is sent to an LLM endpoint for generating the structured persona data.
6. **Similarity Check**: A basic similarity check is performed on the newly generated persona against its references to discard simple copies or highly similar concepts.
7. **Collection**: Valid and unique personas are added to the final pool, which is saved periodically and at the end of the run.

> [!Note]
> Profession weights have been adjusted to U.S. Bureau of Labor Statistics (BLS) data, ensuring a realistic distribution of professions in the generated personas.

## Known Limitations

- **Narrative Depth**: The background and chatting_style descriptions are intentionally brief. They provide a starting point but lack the depth of a fully developed character biography.
- **Generation Patterns**: Despite efforts to ensure novelty, the generation process may fall into subtle patterns or tropes over 5,000 iterations.

UPDATE: Here follows a list of improvements we plan to implement in the next iteration of this dataset:
- **Extreme Personas**: Consistently negative or erratic emotional states, dysfunctional communication patterns, personas that are defined by their impairments or unusual thought processes, rather than just their coping mechanisms. (UPDATE 04/08/2025: Done!)
- **Waaaaaay more seed components**: While we think our professions list is already quite good, we want to expand the list of life contexts, traits, and chatting quirks to allow for more diverse personas.

## Additional Information
### Code and Seed Data

The generation script and seed data can be found on [GitHub](https://github.com/marcodsn/SPB/tree/2508).

### Licensing Information

This dataset is licensed under the CC BY 4.0 License.
The code used to generate the dataset is available under the Apache 2.0 License.

### Citation Information

If you use this dataset in your research, please consider citing it as follows:

```
@misc{marcodsn_2025_SPB2508,
  title     = {Synthetic Persona Bank},
  author    = {Marco De Santis},
  year      = {2025},
  month     = {August},
  url       = {https://huggingface.co/datasets/marcodsn/SPB-2508},
}
```
