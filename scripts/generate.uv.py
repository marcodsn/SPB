#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai>=1.0",
#   "pydantic"
# ]
# ///

import os
import json
import time
import random
from openai import OpenAI
from pydantic import BaseModel

# Ensure the OpenAI-compatible server details are available
# For a local server like llama.cpp, OPENAI_API_BASE is the key variable.
# The API key is often not required for local servers, but the client needs a value.

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:5000/v1")  # Change this to your server's URL if needed

# Other definitions
NUM_REFERENCES = 3  # Number of seed personas to use as references for each generation

# Define your detailed persona schema
class Persona(BaseModel):
    name: str
    username: str | None = None
    age: int
    traits: list[str]
    background: str
    chatting_style: str

# Define the structured JSON output format for the model
# NOTE: This json_schema feature requires a compatible server (e.g., TogetherAI, Anyscale, or modern llama.cpp with grammar support).
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Persona",
        "schema": Persona.model_json_schema(),
        "strict": True,
    },
}

# Initialize the OpenAI client to connect to a compatible server
client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),  # API key is often not required for local servers
)

# Load persona components from the seed data
with open("data/seed/persona_components.json", "r") as f:
    components = json.load(f)

# Begin with a few seed personas
seed_personas = []
with open("data/seed/personas.json", "r") as f:
    seed_personas = json.load(f)

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

def generate_random_username():
    return random.choice(usernames)

def weighted_choice(component_list):
    """Selects one item from a list of {'value': ..., 'weight': ...} objects."""
    values = [item['value'] for item in component_list]
    weights = [item['weight'] for item in component_list]
    return random.choices(values, weights=weights, k=1)[0]

def weighted_sample(component_list, k):
    """Selects k unique items from a list of weighted objects."""
    values = [item['value'] for item in component_list]
    weights = [item['weight'] for item in component_list]
    return random.choices(values, weights=weights, k=k)

# Iteratively grow the persona pool
persona_pool = seed_personas.copy()
target_n = 10000
# IMPORTANT: Change this to your model's identifier as recognized by your server.
# For llama.cpp, it might be the model file path. For a service, it's a specific string.
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
timestamp = int(time.time())
output_filename = f"data/raw/data_{model_name.split("/")[-1]}_{timestamp}.json"

print(f"Starting persona generation. Target: {target_n} personas.")

while len(persona_pool) < target_n:
    # Select the last few personas as reference for generating the next one
    if len(persona_pool) > 0 and len(persona_pool) % 50 == 0:
        # Use the seed personas as references every 50 iterations to guide the model back to our objectives
        references = random.sample(seed_personas, min(len(seed_personas), NUM_REFERENCES))
    else:
        # Use a dynamic lookback range to lessen recent errors propagation
        lookback_range = min(len(persona_pool), 20)
        reference_pool = persona_pool[-lookback_range:]
        references = random.sample(reference_pool, min(len(reference_pool), NUM_REFERENCES))

    profession = weighted_choice(components['professions'])
    life_context = weighted_choice(components['life_contexts'])
    num_traits = random.randint(3, 5)
    selected_traits = list(set(weighted_sample(components['traits'], k=num_traits)))
    chat_quirk = weighted_choice(components['chatting_quirks'])
    age = random.randint(19, 75)

    instruction = (
        "Here are some examples of personas we have already generated. AVOID repeating their themes:\n"
        + "\n".join([json.dumps(p, ensure_ascii=False) for p in references]) +
        f"\n\n---\n"
        f"Your task is to generate a NEW, unique persona by creatively combining the following random elements. Create a believable, specific character who embodies all these aspects. Do not just list the components; weave them into a coherent story.\n\n"
        f"**BUILDING BLOCKS TO COMBINE:**\n"
        f"- **Profession:** {profession}\n"
        f"- **Life Context:** They are currently {life_context}.\n"
        f"- **Core Character Traits:** Should reflect: {', '.join(selected_traits)}.\n"
        f"- **Chat Style Inspiration:** Their style is inspired by this quirk: \"{chat_quirk}\"\n"
        f"- **Target Age:** The persona must be {age} years old, unless completely unreaonable given their profession, in which case you can choose an age yourself.\n\n"
        f"---"
        f"**JSON OUTPUT TASK:**\n"
        f"Create a single JSON object for a persona named '{generate_random_name()}'. "
        f"Follow these rules for the JSON fields:\n"
        f"- **traits:** Choose 3-6 adjectives from the list above that best fit the final character you imagined.\n"
        f"- **background:** A short, specific 1-2 sentence story (≤300 chars) that integrates the profession, context, and age.\n"
        f"- **chatting_style:** A brief description (≤120 chars) of their texting style, directly inspired by the quirk.\n"
        f"The final output must be only the strict JSON object, with no extra text."
    )

    messages = [
        {"role": "system", "content": "You are a creative persona generator. You will create and output only a single, structured persona JSON object, following the provided schema strictly. Do not add any extra text or explanation."},
        {"role": "user", "content": instruction}
    ]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            response_format=response_format,
            max_tokens=512,
            temperature=0.8,
        )

        persona_json_str = response.choices[0].message.content
        new_persona_data = json.loads(persona_json_str)
        persona_obj = Persona(**new_persona_data)

        # Check for similarity to avoid subtle repetitions
        is_too_similar = False
        for p in references:
            if p['background'].split()[0] == persona_obj.background.split()[0] and len(set(p['traits']) & set(persona_obj.traits)) > 2:
                 print(f"⚠️  Skipping similar persona: {persona_obj.name}")
                 is_too_similar = True
                 break

        if not is_too_similar:
            persona_pool.append(persona_obj.model_dump())
            print(f"✅ Added persona '{persona_obj.name}' (total: {len(persona_pool)})")

    except Exception as e:
        print(f"❌ Error generating persona: {e}")

    # Save to file periodically
    if len(persona_pool) % 5 == 0:
        print(f"\n--- Saving progress to {output_filename} ---\n")
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(persona_pool, f, ensure_ascii=False, indent=2)

# Save the final result
print(f"\nTarget of {target_n} reached. Saving final pool to {output_filename}.")
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(persona_pool, f, ensure_ascii=False, indent=2)

print("✅ Script finished.")
