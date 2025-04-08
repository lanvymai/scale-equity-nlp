import argparse
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# Step 1: Load the evaluation dataset
def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Step 2: Load a vision-language model and processor
def load_model(model_name):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model

# Step 3: Preprocess the image
def preprocess_image(image_url):
    response = requests.get(image_url, stream=True)
    image = Image.open(response.raw).convert("RGB")
    return image

# Step 4: Generate predictions
def generate_prediction(processor, model, question, options, image_url):
    # Preprocess the image
    image = preprocess_image(image_url)
    
    # Combine the question with each option to create text inputs
    text_inputs = [f"{question} {opt}" for opt in options]
    
    # Preprocess inputs for the vision-language model
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    
    # Generate predictions
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text  # Text-to-image similarity scores
    probs = logits_per_text.softmax(dim=1)  # Convert to probabilities
    
    # Find the most likely option
    predicted_index = probs.argmax().item()
    return predicted_index

# Step 5: Evaluate the model
def evaluate_model(dataset, processor, model):
    correct = 0
    total = 0
    for entry in dataset:
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        image_url = entry["image"]

        # Generate prediction (get the index of the predicted option)
        predicted_index = generate_prediction(processor, model, question, options, image_url)
        
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
    processor, model = load_model(args.model)
    
    # Evaluate the model
    accuracy = evaluate_model(dataset, processor, model)
    
    # Print the results
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2f}%")