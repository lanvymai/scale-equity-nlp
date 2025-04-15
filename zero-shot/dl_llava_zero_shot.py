import argparse
import json
from PIL import Image
import requests
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, pipeline

# Step 1: Load the evaluation dataset
def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Step 2: Load a vision-language processor & pipeline
def load_pipeline(model_path):
    processor = AutoProcessor.from_pretrained(model_path)

    quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                            )

    pipe = pipeline(task="image-text-to-text", model=model_path, 
                    tokenizer=processor.tokenizer,
                    image_processor=processor.image_processor,
                    model_kwargs={"quantization_config": quantization_config})
    print("pipeline loaded")

    return pipe

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
        image = Image.open(response.raw)
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

# Step 4: Generate predictions
def generate_prediction(pipe, question, options, image_url):
    # Preprocess the image
    image = preprocess_image(image_url)
    if image is None:
        return None  # Return None if the image is invalid

    max_new_tokens = 200
    a, b, c, d = options

    prompt = f'USER: <image>\n{question}\nOPTIONS: {a}, {b}, {c}, {d}\nASSISTANT:'
    print("PROMPT: ", prompt)

    outputs = pipe(image, text=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})

    full_text = outputs[0]['generated_text']
    model_output = full_text.split("ASSISTANT:")[-1].strip()
    print("MODEL PREDICTION: ", model_output)

    return model_output

# Step 5: Evaluate the model
def evaluate_model(dataset, pipe, output_file):
    correct = 0
    total = 0
    results = []  # List to store results for saving

    for entry in dataset:
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        image_url = entry["image"]
        category = entry["category"]

        # Generate prediction (get the index of the predicted option)
        predicted_output = generate_prediction(pipe, question, options, image_url)
        if predicted_output is None:
            continue  # Skip this entry if the image is invalid
        
        # Store the result
        result = {
            "question": question,
            "image": image_url,
            "options": options,
            "model_selection": predicted_output,
            "correct_answer": correct_answer,
            "is_correct": predicted_output == correct_answer
        }
        results.append(result)

        # Compare the predicted option with the correct answer
        if predicted_output == correct_answer:
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

def calculate_accuracy_by_group(evaluation_set, pipe):
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
        

        # Generate prediction (get the index of the predicted option)
        predicted_option = generate_prediction(pipe, question, options, image_url)
        if predicted_option is None:
            continue  # Skip this entry if the image is invalid

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
    parser.add_argument("--output", type=str, default="model_results.json",
                        help="Path to save the model's selection results (default: model_results.json)")    
    parser.add_argument("--debug", action="store_true",
                        help="Use a small dataset during debugging")
    
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset)

    # Use only 10 examples if in debug mode
    if args.debug:
        dataset = dataset[:10]
    
    # Load the pipeline
    model_path = "/teamspace/studios/this_studio/scale-equity-nlp/models/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369"
    pipe = load_pipeline(model_path)
    
    # Evaluate the model
    accuracy = evaluate_model(dataset, pipe, args.output)
    
    # Calculate group accuracy
    group_accuracies = calculate_accuracy_by_group(dataset, pipe)

    # Save the results
    save_group_accuracy(group_accuracies, "llava_group_accuracy_results_debug.json")

    # Print the results
    print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2f}%")