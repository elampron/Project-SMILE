import json
import os

def get_summary_examples(file_path='examples/summary_examples.json'):
    """
    Loads summary examples from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing examples.

    Returns:
        str: A formatted string of examples to be included in the prompt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        examples = json.load(file)
    
    formatted_examples = ""
    for example in examples:
        formatted_examples += f"**Example:**\n\n"
        formatted_examples += "Conversation Data:\n"
        formatted_examples += "```json\n"
        formatted_examples += json.dumps(example['conversation_data'], indent=2)
        formatted_examples += "\n```\n\n"
        formatted_examples += "Expected Structured Output:\n"
        formatted_examples += "```json\n"
        formatted_examples += json.dumps(example['expected_structured_output'], indent=2)
        formatted_examples += "\n```\n\n---\n\n"
    
    return formatted_examples

def get_entity_extraction_examples(file_path='examples/entity_extractor_examples.json'):
    """
    Loads entity extraction examples from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing examples.

    Returns:
        str: A formatted string of examples to be included in the prompt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as file:
        examples = json.load(file)
    
    formatted_examples = ""
    for idx, example in enumerate(examples, start=1):
        formatted_examples += f"**Example {idx}:**\n\n"
        formatted_examples += "Conversation Data:\n"
        formatted_examples += "```json\n"
        formatted_examples += json.dumps(example['conversation_data'], indent=2)
        formatted_examples += "\n```\n\n"
        formatted_examples += "Expected Structured Output:\n"
        formatted_examples += "```json\n"
        formatted_examples += json.dumps(example['expected_structured_output'], indent=2)
        formatted_examples += "\n```\n\n---\n\n"
    
    return formatted_examples

def get_preference_extraction_examples(file_path='examples/preference_extractor_examples.json'):
    """
    Loads preference extraction examples from a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as file:
        examples = file.read()
    
    return examples

def get_cognitive_memory_examples(file_path='examples/cognitive_memory_examples.json'):
    """
    Loads cognitive memory examples from a JSON file.
    Each example shows how to extract cognitive memories from conversations,
    demonstrating different memory types and structures.
    
    Args:
        file_path (str): Path to the JSON file containing examples.
        
    Returns:
        str: A formatted string of examples to be included in the prompt.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    formatted_examples = ""
    for idx, example in enumerate(data['examples'], start=1):
        formatted_examples += f"**Example {idx}:**\n\n"
        
        # Add conversation data
        formatted_examples += "Conversation Data:\n"
        formatted_examples += "```json\n"
        formatted_examples += json.dumps(example['conversation_data'], indent=2)
        formatted_examples += "\n```\n\n"
        
        # Add memory type index
        if 'memory_type_index' in example:
            formatted_examples += "Memory Type Index:\n"
            formatted_examples += "```json\n"
            formatted_examples += json.dumps(example['memory_type_index'], indent=2)
            formatted_examples += "\n```\n\n"
        
        # Add expected cognitive memories
        formatted_examples += "Expected Cognitive Memories:\n"
        formatted_examples += "```json\n"
        formatted_examples += json.dumps(example['expected_cognitive_memories'], indent=2)
        formatted_examples += "\n```\n\n"
        
        formatted_examples += "---\n\n"
    
    return formatted_examples