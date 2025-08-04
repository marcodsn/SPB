#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# ///

import os
import json
import glob
import time
import uuid

# Define paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def extract_model_name_from_filename(filename):
    """
    Extracts the model name from a filename following the pattern:
    data_{model_name}_{timestamp}.json
    Handles model names that might contain underscores.
    """
    base_name = os.path.basename(filename)
    # Remove the 'data_' prefix
    if base_name.startswith("data_"):
        base_name = base_name[len("data_"):]

    # Remove the '_{timestamp}.json' suffix
    # Find the last underscore, which precedes the timestamp
    last_underscore_index = base_name.rfind('_')
    if last_underscore_index != -1:
        # Check if the part after the last underscore is a timestamp followed by .json
        timestamp_part = base_name[last_underscore_index+1:].split('.')[0]
        if timestamp_part.isdigit():
            model_name = base_name[:last_underscore_index]
            return model_name
    return None # Fallback if pattern doesn't match

def get_persona_dedupe_key(persona):
    """
    Creates a unique key for a persona to identify duplicates.
    Uses 'name', 'age', sorted 'traits', 'background', and 'chatting_style'.
    """
    # Ensure traits are sorted to make the key consistent regardless of original order
    traits_tuple = tuple(sorted(persona.get('traits', [])))

    # Use a tuple of immutable fields as the key
    return (
        persona.get('name'),
        persona.get('age'),
        traits_tuple,
        persona.get('background'),
        persona.get('chatting_style')
    )

def process_raw_data():
    """
    Reads all JSON files from the raw directory, merges them,
    adds a 'model' field, removes duplicates, and saves to a processed file.
    """
    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    all_personas = []
    seen_keys = set()
    total_files_processed = 0
    total_personas_loaded = 0

    # Get all JSON files in the raw data directory
    json_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.json"))

    if not json_files:
        print(f"No JSON files found in '{RAW_DATA_DIR}'. Exiting.")
        return

    print(f"Found {len(json_files)} JSON files to process in '{RAW_DATA_DIR}'.")

    for file_path in json_files:
        total_files_processed += 1
        model_name = extract_model_name_from_filename(file_path)

        if not model_name:
            print(f"Warning: Could not extract model name from '{os.path.basename(file_path)}'. Skipping this file.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                personas_from_file = json.load(f)

            if not isinstance(personas_from_file, list):
                print(f"Warning: File '{os.path.basename(file_path)}' does not contain a JSON list. Skipping.")
                continue

            print(f"Processing '{os.path.basename(file_path)}' (Model: {model_name}) with {len(personas_from_file)} personas.")
            total_personas_loaded += len(personas_from_file)

            for persona in personas_from_file:
                # Add the 'model' field
                persona['model'] = model_name
                # Add an 'id' field
                persona['id'] = uuid.uuid4().hex
                all_personas.append(persona)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{os.path.basename(file_path)}': {e}. Skipping file.")
        except Exception as e:
            print(f"An unexpected error occurred with '{os.path.basename(file_path)}': {e}. Skipping file.")

    print("\n--- Merging and Deduplicating ---")
    print(f"Total personas loaded across all files: {total_personas_loaded}")

    unique_personas = []
    for persona in all_personas:
        key = get_persona_dedupe_key(persona)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_personas.append(persona)

    print(f"Total unique personas after deduplication: {len(unique_personas)}")
    print(f"Removed {total_personas_loaded - len(unique_personas)} duplicate(s).")

    # Define the output filename
    timestamp = int(time.time())
    output_filename = os.path.join(PROCESSED_DATA_DIR, f"processed_personas_{timestamp}.jsonl")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for persona in unique_personas:
                f.write(json.dumps(persona, ensure_ascii=False) + '\n')
        print(f"\nSuccessfully saved processed data to '{output_filename}'")
    except Exception as e:
        print(f"Error saving processed data to '{output_filename}': {e}")

if __name__ == "__main__":
    process_raw_data()
