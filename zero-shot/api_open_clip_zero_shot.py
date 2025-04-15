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
    
    except requests.exceptions.RequestException as e:
        print(f"HTTP error while fetching image: {image_url} - {e}")
        return None
    except ValueError as e:
        print(f"Value error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error while processing image: {image_url} - {e}")
        return None
    # response = requests.get(image_url, stream=True)
    # image = Image.open(response.raw).convert("RGB")
    # return image

# Step 4: Generate predictions
def generate_prediction(processor, model, question, options, image_url):
    # Preprocess the image
    image = preprocess_image(image_url)
    if image is None:
        return None  # Return None if the image is invalid

    # Combine the question with each option to create text inputs
    text_inputs = [f"{question} {opt}" for opt in options]
    
    # Preprocess inputs for the vision-language model
    inputs = processor(text=text_inputs, 
                        images=image, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,  # Truncate text inputs
                        max_length=77)    # CLIP's max length is 77 tokens
    # print(f"Processor inputs: {inputs.keys()}")  # Should include 'input_ids' and 'pixel_values'
    # print(f"Text inputs: {text_inputs}")
    # print(f"Input IDs shape: {inputs['input_ids'].shape}")
    # print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    # Generate predictions
    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text  # Text-to-image similarity scores
    probs = logits_per_text.softmax(dim=1)  # Convert to probabilities
    
    # Find the most likely option
    predicted_index = probs.argmax().item()
    return predicted_index

# Step 5: Evaluate the model
def evaluate_model(dataset, processor, model, output_file):
    correct = 0
    total = 0
    results = []  # List to store results for saving

    for entry in dataset:
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        image_url = entry["image"]
        category = entry["category"]

        # image = preprocess_image(image_url)
        # if image is None:
        #     print(f"Skipping entry due to invalid image: {image_url}")
        #     continue

        # Generate prediction (get the index of the predicted option)
        predicted_index = generate_prediction(processor, model, question, options, image_url)
        if predicted_index is None:
            continue  # Skip this entry if the image is invalid
        
        predicted_option = options[predicted_index]

        # Store the result
        result = {
            "question": question,
            "image": image_url,
            "options": options,
            "model_selection": predicted_option,
            "correct_answer": correct_answer,
            "is_correct": predicted_option == correct_answer
        }
        results.append(result)

        # Compare the predicted option with the correct answer
        if options[predicted_index] == correct_answer:
            correct += 1
        total += 1

    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
    
    # Calculate accuracy
    accuracy = correct / total * 100
    return accuracy

# Step 6: Evaluate Group Accuracy
# Define group mappings
western_categories = {
    "Europe", "North America", "Australia", "New Zealand", "Greco-Roman", "Enlightenment",
    "Christian", "Humanist", "Individualism", "Secularism", "Rationalism", "Liberalism",
    "Democracy", "Realism", "Modernism", "Minimalism", "Gothic architecture",
    "Baroque architecture"
}

non_western_categories = {
    "South Asia", "Southeast Asia", "Middle East", "East Asia", "Africa", "Latin America",
    "Indigenous Culture", "Confucian", "Taoist", "Hindu", "Islamic", "Buddhist", "Animist",
    "Collectivism", "Spirituality", "Tradition", "Hierarchy", "Symbolism", "Oral tradition",
    "Islamic architecture", "Buddhist architecture", "Vernacular architecture"
}

continent_categories = {
    "Europe": {"Europe"},
    "North America": {"North America"},
    "Oceania": {"Australia", "New Zealand"},
    "Asia": {"South Asia", "Southeast Asia", "East Asia"},
    "Middle East": {"Middle East"},
    "Africa": {"Africa"},
    "Latin America": {"Latin America"}
}
def calculate_accuracy_by_group(evaluation_set, processor, model):
    group_accuracies = {
        "Western": {"correct": 0, "total": 0},
        "Non-Western": {"correct": 0, "total": 0},
        "Continents": {continent: {"correct": 0, "total": 0} for continent in continent_categories}
    }

    for entry in evaluation_set:
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        image_url = entry["image"]
        category = entry["category"]
        
        # image = preprocess_image(image_url)
        # if image is None:
        #     print(f"Skipping entry due to invalid image: {image_url}")
        #     continue

        # Generate prediction (get the index of the predicted option)
        predicted_index = generate_prediction(processor, model, question, options, image_url)
        if predicted_index is None:
            continue  # Skip this entry if the image is invalid
        predicted_option = options[predicted_index]

        # Western vs. Non-Western
        if category in western_categories:
            group = "Western"
        elif category in non_western_categories:
            group = "Non-Western"
        else:
            continue

        group_accuracies[group]["total"] += 1
        if predicted_option == correct_answer:
            group_accuracies[group]["correct"] += 1

        # Continents
        for continent, categories in continent_categories.items():
            if category in categories:
                group_accuracies["Continents"][continent]["total"] += 1
                if predicted_option == correct_answer:
                    group_accuracies["Continents"][continent]["correct"] += 1

    # Calculate accuracy percentages
    for group in ["Western", "Non-Western"]:
        correct = group_accuracies[group]["correct"]
        total = group_accuracies[group]["total"]
        group_accuracies[group]["accuracy"] = (correct / total * 100) if total > 0 else 0

    for continent, stats in group_accuracies["Continents"].items():
        correct = stats["correct"]
        total = stats["total"]
        stats["accuracy"] = (correct / total * 100) if total > 0 else 0

    return group_accuracies

# Step 7: Save group accuracy results
def save_group_accuracy(group_accuracies, filename):
    with open(filename, 'w') as f:
        json.dump(group_accuracies, f, indent=4)
    print(f"Group accuracy results saved to {filename}")

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluates a vision-language model on Wikipedia Evaluation Dataset "
                    "using a multiple-choice paradigm for a visual question-answering task.")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the Hugging Face model to use (e.g., openai/clip-vit-base-patch32)")
    parser.add_argument("--dataset", type=str, default="evaluation_dataset.json",
                        help="Path to the evaluation dataset (default: evaluation_dataset.json)")
    parser.add_argument("--output", type=str, default="open_clip_results.json",
                        help="Path to save the model's selection results (default: open_clip_results.json)")    
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
    accuracy = evaluate_model(dataset, processor, model, args.output)
    
    # Calculate group accuracy
    group_accuracies = calculate_accuracy_by_group(dataset, processor, model)

    # Save the results
    save_group_accuracy(group_accuracies, "group_accuracy_results_final.json")

    # Print the results
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2f}%")