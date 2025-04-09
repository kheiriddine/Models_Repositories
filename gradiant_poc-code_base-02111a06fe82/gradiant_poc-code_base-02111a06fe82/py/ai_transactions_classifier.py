import boto3
import json
from difflib import get_close_matches

# AWS Bedrock Configuration
AWS_REGION = "us-east-1"  # Replace with your AWS region
BEDROCK_MODEL_ID = "anthropic.claude-v2"  # Replace with the selected model
BEDROCK_CLIENT = boto3.client('bedrock-runtime', region_name=AWS_REGION)

# Predefined categories (provided by the user)
PREDEFINED_CATEGORIES = [
    "Consulting", "Logistics", "IT Equipment", "Raw Materials", "R&D"
]

def read_file(file_path):
    """ Reads a .txt file and returns its content. """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def find_closest_category(text, categories):
    """ Finds the closest matching category from the predefined list. """
    match = get_close_matches(text, categories, n=1, cutoff=0.6)  # Similarity threshold
    return match[0] if match else "Unknown"

def classify_with_bedrock(text):
    """ Sends the text to Amazon Bedrock for classification. """
    prompt = f"""
    Classify the following transaction description into a procurement category:

    Description: {text}

    Choose one from these predefined categories: {", ".join(PREDEFINED_CATEGORIES)}.
    If none fit, suggest the most accurate category.

    Output format (JSON):
    {{
      "predicted_category": "Suggested category"
    }}
    """
    
    response = BEDROCK_CLIENT.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"prompt": prompt})
    )

    result = json.loads(response['body'].read().decode())
    return result.get("predicted_category", "Unknown")

def classify_transaction(file_path):
    """ Classifies a .txt file and returns 2 categories: the closest match and the AI suggestion. """
    text = read_file(file_path)

    closest_category = find_closest_category(text, PREDEFINED_CATEGORIES)
    ai_suggested_category = classify_with_bedrock(text)

    return {"Closest_Category": closest_category, "AI_Suggested_Category": ai_suggested_category}

# Example Usage
if __name__ == "__main__":
    input_file = "transaction.txt"  # Input file
    result = classify_transaction(input_file)
    print(result)
