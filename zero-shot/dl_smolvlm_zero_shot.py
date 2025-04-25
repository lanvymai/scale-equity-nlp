import argparse
import json
from PIL import Image
from tqdm import tqdm  # For progress bars
import time
import requests
import torch
import re
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq

# Step 1: Load the evaluation dataset
def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Step 2: Load a vision-language processor & pipeline
def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model =  AutoModelForVision2Seq.from_pretrained(model_path)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)

    # quantization_config = BitsAndBytesConfig(
    #                         load_in_4bit=True,
    #                         bnb_4bit_compute_dtype=torch.float16,
    #                         bnb_4bit_use_double_quant=True,
    #                         bnb_4bit_quant_type="nf4"
    #                         )

    # pipe = pipeline(task="image-text-to-text", model=model_path, 
    #                 tokenizer=processor.tokenizer,
    #                 image_processor=processor.image_processor,
    #                 model_kwargs={"quantization_config": quantization_config})
    # print("Pipeline loaded")

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

# Step 4: Generate predictions
def generate_prediction(processor, model, question, options, image_url):

    letter_mapping_dct = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # Preprocess the image
    image = preprocess_image(image_url)
    if image is None:
        return None  # Return None if the image is invalid

    max_new_tokens = 200
    a, b, c, d = options

    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "data": image},  # image input
            {"type": "text", "text": f"{question} Provide only the option's letter from the given choice."},  # the question
            {"type": "text", "text": f"OPTIONS:\nA.{a}\nB.{b}\nC.{c}\nD.{d}"}  # options
        ]
    },
    ]
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(DEVICE)

    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # Print the response from the assistant
    # print(generated_texts)
    match = re.search(r'Assistant: (?:Answer: )?(\w)', generated_texts[0])

    # If a match is found, extract the letter (i.e., the answer)
    if match:
        prediction = match.group(1)
        # print(prediction)

    # outputs = pipe(image, text=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})

    # full_text = outputs[0]['generated_text']
    # prediction = full_text.split("ASSISTANT:")[-1].strip()

    if prediction in letter_mapping_dct:
        predicted_index = letter_mapping_dct[prediction]
    else:
        return None

    return predicted_index


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

# Step 5: Evaluate the model & Evaluate Group Accuracy
def evaluate_model(dataset, processor, model, output_file):
    num_correct = 0
    num_q = 0
    results = []  # List to store results for saving

    start_time = time.time()  # Start timing the evaluation

    group_accuracies = {
        "Western": {"correct": 0, "total": 0},
        "Non-Western": {"correct": 0, "total": 0},
        "Continents": {continent: {"correct": 0, "total": 0} for continent in continent_categories}
    }

    for entry in tqdm(dataset, desc="Evaluating entries", ncols=100):
        question = entry["question"]
        options = entry["options"]
        correct_answer = entry["answer"]
        image_url = entry["image"]
        category = entry["category"]

        # Generate prediction (get the index of the predicted option)
        predicted_index = generate_prediction(processor, model, question, options, image_url)
        if predicted_index is None:
            continue  # Skip this entry if the image is invalid

        predicted_output = options[predicted_index]

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
        if predicted_output == correct_answer:
            group_accuracies[group]["correct"] += 1

        # Continents
        for continent, categories in continent_categories.items():
            if category in categories:
                group_accuracies["Continents"][continent]["total"] += 1
                if predicted_output == correct_answer:
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


    elapsed = time.time() - start_time  # Total time elapsed
    # Show minutes and seconds separately
    m, s = divmod(elapsed, 60)
    print(f"Total evaluation time: {int(m)} minutes and {s:.2f} seconds")


    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
    
    # Calculate accuracy
    accuracy = num_correct / num_q * 100

    return accuracy, group_accuracies


# Step 6: Save group accuracy results
def save_group_accuracy(group_accuracies, filename):
    with open(filename, 'w') as f:
        json.dump(group_accuracies, f, indent=4)
    print(f"Group accuracy results saved to {filename}")

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluates a vision-language model on Wikipedia Evaluation Dataset "
                    "using a multiple-choice paradigm for a visual question-answering task.")
    
    # parser.add_argument("--model", type=str, required=True,
    #                     help="Path of the model to use (e.g., openai/clip-vit-base-patch32)")
    parser.add_argument("--dataset", type=str, default="evaluation_dataset.json",
                        help="Path to the evaluation dataset (default: evaluation_dataset.json)")
    parser.add_argument("--output", type=str, default="scale-equity-nlp/output_files/model_results.json",
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
    """ CHANGE MODEL PATH ACCORDINGLY """
    model_path = "/teamspace/studios/this_studio/scale-equity-nlp/models/SmolVLM/models--HuggingFaceTB--SmolVLM-Instruct/snapshots/81cd9a775a4d644f2faf4e7becff4559b46b14c7"
    processor, model = load_model(model_path) # Change back to args.model later
    
    # Evaluate the model & Calculate group accuracy
    accuracy, group_accuracies = evaluate_model(dataset, processor, model, args.output)

    # Save the results
    save_group_accuracy(group_accuracies, "/teamspace/studios/this_studio/scale-equity-nlp/output_files/smolvlm_group_accuracy_results_all.json")

    # Print the results
    # print(f"Model: {args.model}")
    print(f"Accuracy: {accuracy:.2f}%")