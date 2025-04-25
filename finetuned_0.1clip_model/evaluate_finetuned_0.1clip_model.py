import argparse
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import time
from tqdm import tqdm
from safetensors.torch import load_file

# Step 1: Load the evaluation dataset
def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Step 2: Load a vision-language model and processor
def load_model(base_model_path, adapter_path):
    # Load the base model
    model = CLIPModel.from_pretrained(base_model_path)

    # Load the adapter weights
    adapter_weights = load_file(adapter_path)
    model.load_state_dict(adapter_weights, strict=False)

    # Load the processor
    processor = CLIPProcessor.from_pretrained(base_model_path)

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
        return None # Return None if the image is invalid
        #raise ValueError(f"Invalid image at URL: {image_url}")

    # Combine the question with each option to create text inputs
    text_inputs = [f"{question} {opt}" for opt in options]
    
    # Preprocess inputs for the vision-language model
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    # print(f"Processor inputs: {inputs.keys()}")  # Should include 'input_ids' and 'pixel_values'
    
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

# Step 5: Evaluate the model
def evaluate_model(dataset, processor, model, output_file):
    num_correct = 0
    num_q = 0
    results = []  # List to store results for saving
    # correct = 0
    # total = 0
    # output_dct = {}

    start_time = time.time()

    group_accuracies = {
        "Western": {"correct": 0, "total": 0},
        "Non-Western": {"correct": 0, "total": 0},
        "Continents": {continent: {"correct": 0, "total": 0} for continent in continent_categories}
    }

    with tqdm(dataset, desc="Evaluating", unit="item", total=len(dataset)) as pbar:
        for entry in pbar:
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

            # Print the model's selection
            # print(f"Question: {question}")
            # print(f"Image URL: {image_url}")
            # print(f"Options: {options}")
            # print(f"Model's Selection: {predicted_option}")
            # print(f"Correct Answer: {correct_answer}")
            # print("-" * 50)

            # new_key = f"Question {total}"
            # output_dct[new_key] = {
            #     "question": question,  
            #     "image_url": image_url,
            #     "options": options,
            #     "model_prediction": predicted_option,
            #     "correct_answer": correct_answer
            # }

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
            if predicted_option == correct_answer:
                num_correct += 1
            num_q += 1

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

        # update the bar’s postfix with live stats
        elapsed   = time.time() - start_time
        avg_time  = elapsed / num_q
        remaining = avg_time * (len(dataset) - num_q)
        pbar.set_postfix({
            "acc": f"{num_correct/num_q*100:5.2f}%",
            "elapsed": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
            "eta": time.strftime("%H:%M:%S", time.gmtime(remaining))
        })

        # Save results to a file
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
        
        # Calculate accuracy
        accuracy = num_correct / num_q * 100

        #save_outputs(output_dct, accuracy)
        return accuracy, group_accuracies

# Step 6: Save group accuracy results
def save_group_accuracy(group_accuracies, filename):
    with open(filename, 'w') as f:
        json.dump(group_accuracies, f, indent=4)
    print(f"Group accuracy results saved to {filename}")


# Step 6: Main script
if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluates afinetuned vision-language model on Wikipedia Evaluation Dataset "
                    "using a multiple-choice paradigm for a visual question-answering task.")
    
    # parser.add_argument("--model", type=str, required=True,
    #                     help="Name of the Hugging Face model to use (e.g., openai/clip-vit-base-patch32)")
    parser.add_argument("--dataset", type=str, default="evaluation_dataset.json",
                        help="Path to the evaluation dataset")
    parser.add_argument("--output", type=str, default="finetuned_0.1_open_clip_results.json",
                        help="Path to save the model's selection results")    
    parser.add_argument("--debug", action="store_true",
                        help="Use a small dataset during debugging")
    
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset)

    # Use only 10 examples if in debug mode
    if args.debug:
        dataset = dataset[:10]
    
    # Paths to the base model and adapter weights
    base_model_path = "/Users/maingoclanvy/scale-equity-nlp/models/CLIP-ViT-B-32/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    adapter_path = "/Users/maingoclanvy/scale-equity-nlp/finetuned_0.1clip_model/adapter_model.safetensors"

    # Load the model and processor
    processor, model = load_model(base_model_path, adapter_path)
    
    # Evaluate the model
    accuracy, group_accuracies = evaluate_model(dataset, processor, model, args.output)
    # Save the results
    save_group_accuracy(group_accuracies, "finetuned_0.1_open_clip_group_accuracy_results.json")

    # Print the results
    print(f"Model: finetuned CLIP-ViT-B-32 on 0.1% LAION-400m")
    print(f"Accuracy: {accuracy:.2f}%")