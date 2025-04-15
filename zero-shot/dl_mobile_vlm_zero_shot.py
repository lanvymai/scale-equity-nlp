import argparse
import json
from transformers import AutoTokenizer, LlamaForCausalLM
from PIL import Image
import requests
import torch

# Step 1: Load the evaluation dataset
def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Step 2: Load a vision-language model and processor
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# Step 3: Preprocess the image
def preprocess_image(image_url):
    # Ensure the URL starts with "http://" or "https://"
    try:
        # Make the HTTP request
        headers = {'User-Agent': 'NYU-NLU-class-project/0.0 (https://docs.google.com/document/d/1qJJJCK7chA6uXlnENVmzdz8LIj0FSW28-XLyHTKKEAc/edit?usp=sharing; nm4867@nyu.edu)'}
        response = requests.get(image_url, stream=True, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP errors

        # Check if the response is an image
        if "image" not in response.headers.get("Content-Type", ""):
            raise ValueError(f"URL does not point to an image: {image_url}")

        # Open the image
        image = Image.open(response.raw).convert("RGB")
        return image
    
    except Exception as e:
        print(f"Error processing image URL {image_url}: {e}")
        return None

    # response = requests.get(image_url, stream=True)
    # image = Image.open(response.raw).convert("RGB")
    # return image

# Step 4: Generate predictions
def generate_prediction(tokenizer, model, question, options, image_url):
    # Preprocess the image
    image = preprocess_image(image_url)
    if image is None:
        raise ValueError(f"Invalid image at URL: {image_url}")

    # Combine the question with each option to create text inputs
    text_inputs = [f"{question} {opt}" for opt in options]
    
     # Tokenize the text inputs
    inputs = tokenizer(text_inputs, padding=True, truncation=True, return_tensors="pt")
    # # Preprocess image (the image part stays the same)
    # image_inputs = preprocess_image(image)  # Preprocessing image with some specific method
    
    # # Ensure the image inputs are in the correct format for the model (e.g., pixel_values)
    # inputs['pixel_values'] = image_inputs  # Add the preprocessed image inputs
    
    print(f"Tokenizer inputs: {inputs.keys()}")  # Should include 'input_ids' and 'pixel_values'
    
    # Generate predictions
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text  # Text-to-image similarity scores
    probs = logits_per_text.softmax(dim=1)  # Convert to probabilities
    
    # Find the most likely option
    predicted_index = probs.argmax().item()
    return predicted_index


# Step 5: Evaluate the model
def evaluate_model(dataset, tokenizer, model):
    correct = 0
    total = 0
    for entry in dataset:
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        image_url = entry["image"]

        image = preprocess_image(image_url)
        if image is None:
            print(f"Skipping entry due to invalid image: {image_url}")
            continue

        # Generate prediction (get the index of the predicted option)
        predicted_index = generate_prediction(tokenizer, model, question, options, image_url)
        predicted_option = options[predicted_index]

        # Print the model's selection
        print(f"Question: {question}")
        print(f"Image URL: {image_url}")
        print(f"Options: {options}")
        print(f"Model's Selection: {predicted_option}")
        print(f"Correct Answer: {correct_answer}")
        print("-" * 50)

        # Compare the predicted option with the correct answer
        if options[predicted_index] == correct_answer:
            correct += 1
        total += 1
    
    # Calculate accuracy
    accuracy = correct / total * 100
    return accuracy

# Step 6: Main script
if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluates a vision-language model on Wikipedia Evaluation Dataset "
                    "using a multiple-choice paradigm for a visual question-answering task.")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the Hugging Face model to use (e.g., openai/clip-vit-base-patch32)")
    parser.add_argument("--dataset", type=str, default="evaluation_dataset.json",
                        help="Path to the evaluation dataset (default: evaluation_dataset.json)")
    parser.add_argument("--debug", action="store_true",
                        help="Use a small dataset during debugging")
    
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset)

    # Use only 10 examples if in debug mode
    if args.debug:
        dataset = dataset[:10]
    
    # Load the model and processor
    path = '/teamspace/studios/this_studio/scale-equity-nlp/models/MobileVLM_V2-1.7B/models--mtgv--MobileVLM_V2-1.7B/snapshots/9a5b623a83feae6a6b2ecad7a843334ccc119ce1'
    tokenizer, model = load_model(path)
    
    # Evaluate the model
    accuracy = evaluate_model(dataset, tokenizer, model)
    
    # Print the results
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2f}%")