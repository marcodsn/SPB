#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface-hub>=0.23",
#   "pydantic"
# ]
# ///

import os
import json
import time
import random
import asyncio
from typing import List, Set
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel

# --- Configuration ---
TARGET_N = 5000    # Target number of unique personas to generate
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"  #  "zai-org/GLM-4.5-Air-FP8"  # "moonshotai/Kimi-K2-Instruct" # Or any other compatible model
CONCURRENCY = 20      # Max number of parallel requests.
CHECKPOINT_EVERY = 50  # Save progress to the file after this many successful generations.
RESET_EVERY = 50      # Reset to seed personas every X generations to prevent drift.
NUM_REFERENCES = 3    # Number of seed personas to use as references for each generation

# Ensure the Hugging Face token is available
if "HF_TOKEN" not in os.environ:
    raise ValueError("Missing Hugging Face token. Please set the HF_TOKEN environment variable.")

# Use the Asynchronous client for parallel execution.
# The provider can be changed to "huggingface", "anyscale", etc.
client = AsyncInferenceClient(
    provider="together",
    # provider="fireworks-ai",
    api_key=os.environ.get("HF_TOKEN")
)

# --- Pydantic Models for Data Structure ---

# Define your detailed persona schema
class Persona(BaseModel):
    name: str
    username: str | None = None
    age: int
    traits: list[str]
    background: str
    chatting_style: str

# --- API Configuration ---

# Define the structured JSON output format for the model
# NOTE: This json_schema feature requires a compatible server.
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Persona",
        "schema": Persona.model_json_schema(),
        "strict": True,
    },
}

# --- Data Loading and Helper Functions ---

# Load persona components from the seed data
with open("data/seed/persona_components.json", "r") as f:
    components = json.load(f)

# Begin with a few seed personas
seed_personas = []
with open("data/seed/personas.jsonl", "r") as f:
    seed_personas = [json.loads(line) for line in f]

# Seed names for the personas
first_names = []
with open("data/seed/first_names.json", "r") as f:
    first_names = json.load(f)

last_names = []
with open("data/seed/last_names.json", "r") as f:
    last_names = json.load(f)

# Example usernames to guide the model
usernames = []
with open("data/seed/usernames.json", "r") as f:
    usernames = json.load(f)

def generate_random_name():
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    return f"{first_name} {last_name}"

def weighted_choice(component_list):
    """Selects one dictionary item from a list of {'value': ..., 'weight': ...} objects."""
    weights = [item['weight'] for item in component_list]
    return random.choices(component_list, weights=weights, k=1)[0]

def weighted_sample(component_list, k):
    """Selects k unique items from a list of weighted objects."""
    values = [item['value'] for item in component_list]
    weights = [item['weight'] for item in component_list]
    return random.choices(values, weights=weights, k=k)

def save_to_jsonl(data: dict, filepath: str):
    """Appends a dictionary as a new line in a JSONL file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# --- Main Generation Logic (Parallelized) ---

async def generate_one_persona(
    semaphore: asyncio.Semaphore,
    reference_personas: List[dict]
) -> Persona | None:
    """Generates a single new persona using the LLM."""
    async with semaphore:
        try:
            # --- Tactic 1: Constrain inputs before the prompt ---
            # 1. Pick the profession FIRST. This returns the full dictionary.
            profession_data = weighted_choice(components['professions'])
            profession = profession_data['value'] # Extract the string value

            # 2. Determine the valid age range based on the profession's metadata
            min_age_for_job = profession_data.get('min_age', 19)
            max_age_for_job = profession_data.get('max_age', 75)
            age = random.randint(min_age_for_job, max_age_for_job)

            # 3. Select other components and immediately extract their 'value'
            # This is the key change to fix the logic for these variables.
            life_context = weighted_choice(components['life_contexts'])['value']
            chat_quirk = weighted_choice(components['chatting_quirks'])['value']

            num_traits = 6
            # weighted_sample already returns a list of strings, so it's fine.
            selected_traits = list(set(weighted_sample(components['traits'], k=num_traits)))


            # --- Tactic 2: Use an improved prompt to force reconciliation ---
            instruction = (
                "Here are some examples of personas we have already generated. AVOID repeating their themes or being too similar:\n"
                + "\n".join([json.dumps(p, ensure_ascii=False) for p in reference_personas]) +
                f"\n\n---\n"
                f"Your task is to generate a NEW, unique, and believable persona. You must creatively synthesize the following building blocks into a single, coherent character. Don't just list the components; make them feel like a real person.\n\n"
                f"**BUILDING BLOCKS TO SYNTHESIZE:**\n"
                f"- **Profession:** {profession}\n"
                f"- **Core Demographics:** They are {age} years old.\n"
                f"- **Life Context:** They are currently {life_context}.\n"
                f"- **Personality Profile:** Their character should reflect these traits: {', '.join(selected_traits)}.\n"
                f"- **Chat Style Challenge:** Their base communication style is: \"{chat_quirk}\"\n\n"
                f"---"
                f"**JSON OUTPUT TASK:**\n"
                f"Create a single JSON object for a persona named '{generate_random_name()}'. "
                f"Follow these rules for the JSON fields:\n"
                f"- **traits:** Choose 3-6 adjectives from the list above that best fit the final, integrated character you imagined.\n"
                f"- **background:** A short, specific 1-2 sentence story (≤300 chars) that **synthesizes** the profession, age, and life context into a believable narrative.\n"
                f"- **chatting_style:** A brief description (≤120 chars) of their texting style. **Crucially, explain HOW a {age}-year-old {profession} would adapt or interpret the '{chat_quirk}'**. For example, would they use it ironically, incorrectly, or perfectly? Make it fit their character.\n"
                f"The final output must be only the strict JSON object, with no extra text."
            )

            messages = [
                {"role": "system", "content": "You are a creative persona generator. You will create and output only a single, structured persona JSON object, following the provided schema strictly. Do not add any extra text or explanation."},
                {"role": "user", "content": instruction}
            ]

            response = await client.chat_completion(
                messages=messages,
                model=MODEL_NAME,
                response_format=response_format,
                max_tokens=512,
                temperature=0.8,
            )

            persona_data = json.loads(response.choices[0].message.content)
            return Persona(**persona_data)

        except Exception as e:
            # This is where your error was being printed.
            print(f"⚠️ Error generating a persona: {e}. Retrying with another task.")
            await asyncio.sleep(2) # Prevent rapid-fire failures
            return None


async def main():
    # This pool will grow with new generations and is used for reference selection
    persona_pool = seed_personas.copy()

    if not seed_personas:
        print("⚠️ No seed personas loaded. Cannot proceed with few-shot prompting.")
        return

    timestamp = int(time.time())
    output_filename = f"data/raw/data_{MODEL_NAME.split('/')[-1]}_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    print(f"\nStarting persona generation. Target: {TARGET_N} personas.")
    print(f"Concurrency: {CONCURRENCY}, Checkpoint: {CHECKPOINT_EVERY}, Anti-Drift Reset: {RESET_EVERY}")
    print(f"Output will be saved to: {output_filename}")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks: Set[asyncio.Task] = set()
    results_buffer: List[Persona] = []
    successful_generations = 0

    while successful_generations < TARGET_N:
        # Launch new tasks if we have capacity
        while len(tasks) < CONCURRENCY and successful_generations + len(tasks) < TARGET_N:
            current_iteration = successful_generations + len(tasks)
            # Determine which pool of examples to use for this task
            if RESET_EVERY > 0 and current_iteration > 0 and current_iteration % RESET_EVERY == 0:
                reference_pool = random.sample(seed_personas, min(len(seed_personas), NUM_REFERENCES))
                print(f"--- 🔄 Iteration {current_iteration}: Resetting reference pool to seeds to prevent drift ---")
            else:
                lookback_range = min(len(persona_pool), 20)
                dynamic_reference_pool = persona_pool[-lookback_range:]
                reference_pool = random.sample(dynamic_reference_pool, min(len(dynamic_reference_pool), NUM_REFERENCES))

            task = asyncio.create_task(
                generate_one_persona(semaphore, reference_pool)
            )
            tasks.add(task)

        # Wait for the next task to complete
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for future in done:
            result = await future
            tasks.remove(future) # Remove the completed task from the active set

            if result:
                successful_generations += 1
                # Add the new persona to the dynamic pool for future generations
                persona_dict = result.model_dump()
                persona_pool.append(persona_dict)
                results_buffer.append(persona_dict)

                print(f"✅ ({successful_generations}/{TARGET_N}) Generated: {result.name}")

                # Checkpoint logic
                if successful_generations > 0 and successful_generations % CHECKPOINT_EVERY == 0 and results_buffer:
                    print(f"\n--- CHECKPOINT: Saving {len(results_buffer)} personas to {output_filename}... ---")
                    for p_dict in results_buffer:
                        save_to_jsonl(p_dict, output_filename)
                    results_buffer.clear()
                    print("--- ✅ CHECKPOINT Complete. ---\n")

    # Final save for any remaining items in the buffer
    if results_buffer:
        print(f"\n--- FINAL SAVE: Saving {len(results_buffer)} remaining personas to file... ---")
        for p_dict in results_buffer:
            save_to_jsonl(p_dict, output_filename)
        print("--- ✅ FINAL SAVE Complete. ---")

    print("\n-----------------------------------------")
    print(f"✅ Target of {TARGET_N} attempted. {successful_generations} personas successfully generated.")
    print(f"Final data saved in {output_filename}")
    print("-----------------------------------------")

if __name__ == "__main__":
    asyncio.run(main())
