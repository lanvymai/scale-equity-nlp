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
def load_model(model_path):
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    return processor, model

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


# Step 4: Generate predictions
def generate_prediction(processor, model, question, options, image_url):
    # Preprocess the image
    image = preprocess_image(image_url)
    if image is None:
        raise ValueError(f"Invalid image at URL: {image_url}")

    # Combine the question with each option to create text inputs
    text_inputs = [f"{question} {opt}" for opt in options]
    
    # Preprocess inputs for the vision-language model
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    print(f"Processor inputs: {inputs.keys()}")  # Should include 'input_ids' and 'pixel_values'
    
    # Generate predictions
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text  # Text-to-image similarity scores
    probs = logits_per_text.softmax(dim=1)  # Convert to probabilities
    
    # Find the most likely option
    predicted_index = probs.argmax().item()
    return predicted_index


def save_outputs(output_data, accuracy, filename="outputs.json"):
    # Save the outputs to a JSON file
    with open("outputs.json", "a") as f:
        json.dump(output_data, f, indent=4)
        f.write("\n")  # Write a newline for each entry

        # Save the accuracy in same file
        f.write(f"Accuracy: {accuracy:.2f}%\n")

# Step 5: Evaluate the model
def evaluate_model(dataset, processor, model):
    correct = 0
    total = 0
    output_dct = {}

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
        predicted_index = generate_prediction(processor, model, question, options, image_url)
        predicted_option = options[predicted_index]

        # Print the model's selection
        print(f"Question: {question}")
        print(f"Image URL: {image_url}")
        print(f"Options: {options}")
        print(f"Model's Selection: {predicted_option}")
        print(f"Correct Answer: {correct_answer}")
        print("-" * 50)

        new_key = f"Question {total}"
        output_dct[new_key] = {
            "question": question,  
            "image_url": image_url,
            "options": options,
            "model_prediction": predicted_option,
            "correct_answer": correct_answer
        }
        
        # Compare the predicted option with the correct answer
        if options[predicted_index] == correct_answer:
            correct += 1
        total += 1
  
    # Calculate accuracy
    accuracy = correct / total * 100

    save_outputs(output_dct, accuracy)

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
    path = '/teamspace/studios/this_studio/scale-equity-nlp/models/CLIP-ViT-L-14/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
    processor, model = load_model(path)
    
    # Evaluate the model
    accuracy = evaluate_model(dataset, processor, model)
    
    # Print the results
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2f}%")