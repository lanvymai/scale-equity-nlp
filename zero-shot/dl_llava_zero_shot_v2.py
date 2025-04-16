import argparse
import json
from PIL import Image
from tqdm import tqdm
import time
import requests
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, pipeline
from datasets import Dataset

###############
# Step 1: Data Loading
###############
def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

###############
# Step 2: Load a vision-language processor & pipeline
###############
def load_pipeline(model_path):
    processor = AutoProcessor.from_pretrained(model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    pipe = pipeline(
        task="image-text-to-text",
        model=model_path, 
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        model_kwargs={"quantization_config": quantization_config}
    )
    print("Pipeline loaded")
    return pipe

###############
# Step 3: Preprocess Image URL into a PIL Image
###############
def fetch_image(image_url):
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

def add_image_field(example):
    # Add a new key "pil_image" to each example with the loaded image (or None)
    image = fetch_image(example["image"])
    example["pil_image"] = image
    return example

###############
# Step 4: Batch processing function for evaluation
###############
def process_batch(batch):
    """
    This function expects the following keys in each batch:
      - question: list of question strings
      - options: list of lists (each inner list is the 4 options)
      - answer: list of correct answer strings
      - pil_image: list of PIL.Images (or None if image failed to load)
      - image: list of image URLs (for fallback check)
    It returns a dict with a new key "predicted_index" corresponding to the prediction.
    """
    letter_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    questions = batch["question"]
    options_list = batch["options"]
    images = batch["pil_image"]
    image_urls = batch["image"]

    prompts = []
    valid_indices = []  # keep track of indices for which we have valid images

    # Build prompts for examples with valid images.
    for i, image in enumerate(images):
        if image is not None:
            # Assume options are always of length 4; unpack them:
            a, b, c, d = options_list[i]
            prompt = (
                f"USER: <image>\n"
                f"{questions[i]} Answer with the option's letter from the given choices directly.\n"
                f"OPTIONS:\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nASSISTANT:"
            )
            prompts.append(prompt)
            valid_indices.append(i)
        else:
            # For examples with no valid image, we output None later.
            prompts.append(None)

    # Create a list of images and prompts only for valid ones.
    images_for_pipe = []
    prompts_for_pipe = []
    for i in valid_indices:
        images_for_pipe.append(images[i])
        prompts_for_pipe.append(prompts[i])

    # Call the pipeline in batch mode only if we have at least one valid image.
    predictions = [None] * len(questions)
    if images_for_pipe:
        outputs = pipe(images_for_pipe, text=prompts_for_pipe, generate_kwargs={"max_new_tokens": 200})
        # Map each output back to the original index.
        for idx, out in zip(valid_indices, outputs):
            full_text = out['generated_text']
            # Parse the generated text (assume the output is in the form ... "ASSISTANT: X")
            pred_letter = full_text.split("ASSISTANT:")[-1].strip()[0]  # first letter of answer
            # If the letter is not one of A, B, C, D, return None
            if pred_letter in letter_mapping:
                predictions[idx] = letter_mapping[pred_letter]
            else:
                predictions[idx] = None
    else:
        # If none valid, predictions stay as None
        predictions = [None] * len(questions)

    return {"predicted_index": predictions}

###############
# Step 5: Batch Evaluation Function using the dataset map
###############
def evaluate_model_batched(dataset, pipe, output_file, batch_size=4):
    # First, use .map() to add a PIL image to each record.
    dataset = dataset.map(add_image_field)

    # Then, use batch mapping to get predictions.
    dataset = dataset.map(process_batch, batched=True, batch_size=batch_size)

    # Now compute accuracy and collect results.
    num_correct = 0
    num_total = 0
    results = []
    for example in tqdm(dataset, desc="Gathering results", ncols=100):
        predicted_index = example.get("predicted_index")
        # If prediction was not made, skip this example.
        if predicted_index is None:
            continue
        options = example["options"]
        predicted_option = options[predicted_index]
        correct_answer = example["answer"]
        result = {
            "question": example["question"],
            "image": example["image"],
            "options": options,
            "model_selection": predicted_option,
            "correct_answer": correct_answer,
            "is_correct": (predicted_option == correct_answer)
        }
        results.append(result)
        num_total += 1
        if predicted_option == correct_answer:
            num_correct += 1

    accuracy = (num_correct / num_total * 100) if num_total > 0 else 0

    # Save results to a file.
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

    return accuracy

###############
# Step 6: Save group accuracy (unchanged)
###############
def save_group_accuracy(group_accuracies, filename):
    with open(filename, 'w') as f:
        json.dump(group_accuracies, f, indent=4)
    print(f"Group accuracy results saved to {filename}")

###############
# Main script with argument parsing
###############
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates a vision-language model on a Wikipedia Evaluation Dataset "
                    "using a multiple-choice paradigm with batched processing."
    )
    parser.add_argument("--dataset", type=str, default="evaluation_dataset.json",
                        help="Path to the evaluation dataset (default: evaluation_dataset.json)")
    parser.add_argument("--output", type=str, default="model_results.json",
                        help="Path to save the model's prediction results")
    parser.add_argument("--debug", action="store_true",
                        help="Use a small dataset for debugging")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing (default: 4)")

    args = parser.parse_args()
    start_time = time.time() 

    # Load dataset from JSON and convert it to a Hugging Face Dataset.
    raw_data = load_dataset(args.dataset)
    dataset = Dataset.from_list(raw_data)

    if args.debug:
        dataset = dataset.select(range(min(10, len(dataset))))  # use only 10 examples for debugging

    # Change the model_path accordingly.
    model_path = "/teamspace/studios/this_studio/scale-equity-nlp/models/llava-v1.5-7b/models--llava-hf--llava-1.5-7b-hf/snapshots/6ceb2ed33cb8f107a781c431fe2e61574da69369"
    pipe = load_pipeline(model_path)

    # Evaluate using batched processing.
    accuracy = evaluate_model_batched(dataset, pipe, args.output, batch_size=args.batch_size)
    print(f"Model accuracy: {accuracy:.2f}%")

    elapsed = time.time() - start_time  # Total time elapsed
    # Show minutes and seconds separately
    m, s = divmod(elapsed, 60)
    print(f"Total time: {int(m)} minutes and {s:.2f} seconds")
