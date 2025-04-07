import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import random
import os

## Write remove duplicates strings
def remove_duplicates(strings):
    unique_counts = {}
    for s in strings:
        unique_counts[s] = unique_counts.get(s, 0) + 1
    unique_strings = [s for s in strings if unique_counts[s] == 1]
    return unique_strings

# Function to scrape links from a category page
def scrape_cate_for_links(cate_link):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    page = session.get(cate_link)
    bs = BeautifulSoup(page.content, 'html.parser')

    web_links = [] # get link in web
    for j in bs.find_all('div', {'class': 'mw-category-group'}):
        for link in j.find_all('a'):
            web_links.append(link.get('href'))

    web_links = remove_duplicates(web_links) # remove duplicate links

    return web_links # list of link

# Function to scrape image-caption data from a Wikipedia page
def get_data(link):
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    page = session.get(f'https://en.wikipedia.org{link}')
    bs = BeautifulSoup(page.content, 'html.parser')

    result = []
    for i in bs.find_all('figure', {'typeof': 'mw:File/Thumb'}):
        img = {"link_img": "", "caption": ""}
        for j in i.find_all('a'):
            for li in j.find_all('img'):
                img['link_img'] = li.get('src')
        for a in i.find_all('figcaption'):
            img['caption'] = a.text.strip()
        result.append(img)
    return result

# Step 1 & 2: Scrape multiple categories
def scrape_multiple_categories(categories):
    dataset = {}
    for category, link in categories.items():
        print(f"Scraping category: {category}")
        links = scrape_cate_for_links(link)
        category_data = []
        for page_link in links:
            try:
                data = get_data(page_link)
                category_data.extend(data)
            except Exception as e:
                print(f"Error processing link {page_link}: {e}")
        dataset[category] = category_data
    return dataset

# Step 3: Save dataset in JSON format
def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset saved to {filename}")

# Step 4: Create evaluation dataset
def create_evaluation_dataset(dataset):
    evaluation_set = []
    question_templates = [
        "What does this image show?",
        "What is this a picture of?",
        "Which of the following best describes this image?",
        "What is the subject of this photo?",
        "Which caption best matches this image?",
        "Choose the correct description for this image.",
        "Which option is the most accurate label for this photo?",
        "Select the label that is most likely the topic of this image.",
        "Which label is most visually consistent with the image shown?"
    ]
    for category, images in dataset.items():
        for img in images:
            question = random.choice(question_templates)
            correct_caption = img['caption']
            other_captions = [i['caption'] for i in images if i != img]
            random_captions = random.sample(other_captions, min(3, len(other_captions)))
            options = [correct_caption] + random_captions
            random.shuffle(options)
            evaluation_set.append({
                "question": question,
                "image": img['link_img'],
                "options": options,
                "answer": correct_caption,
                "category": category
            })
    return evaluation_set

# Step 5: Save evaluation dataset
def save_evaluation_dataset(evaluation_set, filename):
    with open(filename, 'w') as f:
        json.dump(evaluation_set, f, indent=4)
    print(f"Evaluation dataset saved to {filename}")

# Main script execution
if __name__ == "__main__":
    # Define categories and their links
    categories = {
        "Indigenous Culture": "https://en.wikipedia.org/wiki/Category:Indigenous_culture",
        "Culture of South Asia": "https://en.wikipedia.org/wiki/Category:Culture_of_South_Asia",
        "Culture of Europe": "https://en.wikipedia.org/wiki/Category:Culture_of_Europe",
        "Culture of Africa": "https://en.wikipedia.org/wiki/Category:Culture_of_Africa",
        "Culture of Latin America": "https://en.wikipedia.org/wiki/Category:Culture_of_Latin_America_by_country",
        "Culture of East Asia": "https://en.wikipedia.org/wiki/Category:Culture_of_East_Asia",
        "Culture of Southeast Asia": "https://en.wikipedia.org/wiki/Category:Culture_of_Southeast_Asia",
        "Culture of Middle East": "https://en.wikipedia.org/wiki/Category:Culture_of_the_Middle_East",
        "Gothic architecture": "https://en.wikipedia.org/wiki/Category:Gothic_architecture",
        "Baroque architecture": "https://en.wikipedia.org/wiki/Category:Baroque_architecture",
        "Islamic architecture": "https://en.wikipedia.org/wiki/Category:Islamic_architecture",
        "Buddhist architecture": "https://en.wikipedia.org/wiki/Category:Buddhist_architecture",
        "Vernacular architecture": "https://en.wikipedia.org/wiki/Category:Vernacular_architecture",
        "Holidays": "https://en.wikipedia.org/wiki/Category:Public_holidays_by_country",
    }

    # Step 1 & 2: Scrape data
    dataset = scrape_multiple_categories(categories)

    # Step 3: Save dataset
    save_dataset(dataset, "image_caption_dataset.json")

    # Step 4: Create evaluation dataset
    evaluation_set = create_evaluation_dataset(dataset)

    # Step 5: Save evaluation dataset
    save_evaluation_dataset(evaluation_set, "evaluation_dataset.json")