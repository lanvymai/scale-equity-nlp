import requests
from bs4 import BeautifulSoup
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tqdm
import time
from random import choice

# List of free proxy servers (Example)
proxies = [
    "http://45.77.67.34:3128",
    "http://198.50.152.64:23500",
    "http://103.216.82.198:6667"
]

## Function to get a random proxy
def get_proxy():
    return {"http": choice(proxies), "https": choice(proxies)}

## Write remove duplicates strings
def remove_duplicates(strings):
    unique_counts = {}
    for s in strings:
        unique_counts[s] = unique_counts.get(s, 0) + 1
    unique_strings = [s for s in strings if unique_counts[s] == 1]
    return unique_strings

def scrape_cate_for_links(cate_link):
  ## send request to Wiki
  session = requests.Session()
  retry = Retry(connect = 3, backoff_factor = 0.5)
  adapter = HTTPAdapter(max_retries = retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)
  page = session.get(cate_link)
  bs = BeautifulSoup(page.content)

  # get link in web
  web_link = []
  for j in bs.find_all('div',{'class':'mw-category-group'}):
    for linkss in j.find_all('a'):
        web_link.append(linkss.get('href'))
  # return list of link
  return web_link


## Get data in json format
def get_data(link):
  session = requests.Session()
  retry = Retry(connect = 3, backoff_factor = 0.5)
  adapter = HTTPAdapter(max_retries = retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)
  page = session.get(f'https://en.wikipedia.org{link}')
  bs = BeautifulSoup(page.content)

  result = []
  img =  { "link_img":"", "caption":""}
  for i in bs.find_all('figure',{'typeof':'mw:File/Thumb'}):
    img =  { "link_img":"", "caption":""}
    for j in i.find_all('a'):
      for li in j.find_all('img'):
        img['link_img'] = li.get('src')
    for a in i.find_all('figcaption'):
      img['caption'] = a.text
    result.append(img)
  return result

## 3. Main script execution
category_link = "https://en.wikipedia.org/wiki/Category:Culture_of_India" ## replace this with links of other categories
links = scrape_cate_for_links(category_link)

with open('image_data.txt', 'w') as f:
    for link in links:
        try:
            data = get_data(link)
            for item in data:
                f.write(f"Image Link: {item['link_img']}\n")
                f.write(f"Caption: {item['caption']}\n\n")
                print(f"Image Link: {item['link_img']}")
                print(f"Caption: {item['caption']}\n")
        except Exception as e:
            print(f"Error processing link {link}: {e}")
            f.write(f"Error processing link {link}: {e}\n")