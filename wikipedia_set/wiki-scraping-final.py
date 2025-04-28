import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import random
import os

dimensions = {
    "geographic origin": {
        "Indigenous Culture", "South Asia", "Europe", "North America",
        "Africa", "Latin America", "East Asia", "Southeast Asia",
        "Middle East", "Australia", "New Zealand"
    },
    "aesthetics / architecture": {
         "Realism", "Modernism", "Minimalism", "Gothic architecture", "Baroque architecture", "Symbolism",
         "Oral tradition", "Islamic architecture", "Buddhist architecture", "Vernacular architecture"
     },
    "philosophical roots": {
        "Greco-Roman", "Enlightenment", "Christian", "Humanist", "Confucian", "Taoist", "Hindu", "Islamic",
        "Buddhist", "Animist"
     },
     "cultural values": {
         "Individualism", "Secularism", "Rationalism", "Liberalism", "Democracy", "Collectivism", "Spirituality",
         "Tradition", "Hierarchy"
    }
}

# Define group mappings
western_categories = {
    "Europe", "North America", "Australia", "New Zealand"
    ,"Greco-Roman", "Enlightenment", "Christian", "Humanist"
     , 
     "Individualism", "Secularism", "Rationalism", "Liberalism", "Democracy",
     "Realism", "Modernism", "Minimalism", "Gothic architecture", "Baroque architecture"
}

non_western_categories = {
    "South Asia", "Southeast Asia", "Middle East", "East Asia", "Africa", "Latin America", "Indigenous Culture" 
    ,"Confucian", "Taoist", "Hindu", "Islamic", "Buddhist", "Animist"
     , 
     "Collectivism", "Spirituality", "Tradition", "Hierarchy", 
     "Symbolism", "Oral tradition", "Islamic architecture", "Buddhist architecture", "Vernacular architecture"
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

# Determine the dimension of a category
def get_dimension(category):
    for dimension, categories in dimensions.items():
        if category in categories:
            return dimension
    return "unknown"

# Function to determine if a category is Western or Non-Western
def is_western(category):
    if category in western_categories:
        return "western"
    elif category in non_western_categories:
        return "non-western"
    return "unknown"


# Write remove duplicates strings
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

    # web_links = remove_duplicates(web_links) # remove duplicate links
    # return web_links # list of link
    return list(set(web_links))

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
                if img['link_img'].startswith("//"):
                    img['link_img'] = "https:" + img['link_img']
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
                for img in data:
                    img["category"] = category
                    img["dimension"] = get_dimension(category)
                    img["western"] = is_western(category)
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
                "category": category,
                "dimension": img["dimension"],
                "western": img["western"]
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
        # Geographic origin
        "Indigenous Culture": "https://en.wikipedia.org/wiki/Category:Indigenous_culture",
        "South Asia": "https://en.wikipedia.org/wiki/Category:Culture_of_South_Asia",
        "Europe": "https://en.wikipedia.org/wiki/Category:Culture_of_Europe",
        "North America": "https://en.wikipedia.org/wiki/Category:North_America",
        "Africa": "https://en.wikipedia.org/wiki/Category:Culture_of_Africa",
        "Latin America": "https://en.wikipedia.org/wiki/Category:Culture_of_Latin_America",
        "East Asia": "https://en.wikipedia.org/wiki/Category:Culture_of_East_Asia",
        "Southeast Asia": "https://en.wikipedia.org/wiki/Category:Culture_of_Southeast_Asia",
        "Middle East": "https://en.wikipedia.org/wiki/Category:Culture_of_the_Middle_East",
        "Australia": "https://en.wikipedia.org/wiki/Category:Culture_of_Australia",
        "New Zealand": "https://en.wikipedia.org/wiki/Category:Culture_of_New_Zealand",        
        # Aesthetics / Architecture
        "Realism": "https://en.wikipedia.org/wiki/Category:Realism",
        "Modernism": "https://en.wikipedia.org/wiki/Category:Modernism",
        "Minimalism": "https://en.wikipedia.org/wiki/Category:Minimalism",
        "Gothic architecture": "https://en.wikipedia.org/wiki/Category:Gothic_architecture",
        "Baroque architecture": "https://en.wikipedia.org/wiki/Category:Baroque_architecture",
        "Symbolism": "https://en.wikipedia.org/wiki/Category:Symbolism",
        "Oral tradition": "https://en.wikipedia.org/wiki/Category:Oral_tradition",
        "Islamic architecture": "https://en.wikipedia.org/wiki/Category:Islamic_architecture",
        "Buddhist architecture": "https://en.wikipedia.org/wiki/Category:Buddhist_architecture",
        "Vernacular architecture": "https://en.wikipedia.org/wiki/Category:Vernacular_architecture",
        # Philosophical Roots
        "Greco-Roman": "https://en.wikipedia.org/wiki/Category:Greco-Roman_world",
        "Enlightenment": "https://en.wikipedia.org/wiki/Category:Age_of_Enlightenment",
        "Christian": "https://en.wikipedia.org/wiki/Category:Christian_tradition",
        "Humanist": "https://en.wikipedia.org/wiki/Category:Humanists",
        "Confucian": "https://en.wikipedia.org/wiki/Category:Confucianism",
        "Taoist": "https://en.wikipedia.org/wiki/Category:Taoism",
        "Hindu": "https://en.wikipedia.org/wiki/Category:Hinduism",
        "Islamic": "https://en.wikipedia.org/wiki/Category:Islam",
        "Buddhist": "https://en.wikipedia.org/wiki/Category:Buddhism",
        "Animist": "https://en.wikipedia.org/wiki/Category:Animists",
        # Cultural Values
        "Individualism": "https://en.wikipedia.org/wiki/Category:Individualism",
        "Secularism": "https://en.wikipedia.org/wiki/Category:Secularism",
        "Rationalism": "https://en.wikipedia.org/wiki/Category:Rationalism",
        "Liberalism": "https://en.wikipedia.org/wiki/Category:Liberalism",
        "Democracy": "https://en.wikipedia.org/wiki/Category:Democracy",
        "Collectivism": "https://en.wikipedia.org/wiki/Category:Collectivism",
        "Spirituality": "https://en.wikipedia.org/?title=Category:Spirituality&from=B",
        "Tradition": "https://en.wikipedia.org/wiki/Category:Tradition",
        "Hierarchy": "https://en.wikipedia.org/wiki/Category:Hierarchy"
    }

    # Step 1 & 2: Scrape data
    dataset = scrape_multiple_categories(categories)

    # Step 3: Save dataset
    save_dataset(dataset, "image_caption_dataset_p1.json")

    # Step 4: Create evaluation dataset
    evaluation_set = create_evaluation_dataset(dataset)

    # Step 5: Save evaluation dataset
    save_evaluation_dataset(evaluation_set, "evaluation_dataset_p1.json")
