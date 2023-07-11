# standard imports
import pandas as pd


# webpage scraping imports
import requests
from requests import get
from bs4 import BeautifulSoup
from pprint import pprint
import time
import os


'''
*------------------*
|                  |
|     ACQUIRE      |
|                  |
*------------------*
'''
# ------------------------------------------------------------------------------------
def get_blog_articles():
    """
    Scrape the blog articles from the given URL and return them as a pandas DataFrame.

    Each row in the DataFrame represents one article, with columns for the title and content of the article.

    Parameters:
    url (str): The URL of the blog page to scrape.

    Returns:
    df_articles (DataFrame): A DataFrame with the title and content of each article.
    """
    # Define the base URL
    url = 'https://codeup.com/blog/'
    
    # Remove the maximum column width limit
    pd.set_option('display.max_colwidth', None)

    headers = {"User-Agent": "Chrome/91.0.4472.124"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    blog_articles = []

    # Find the article links
    article_links = soup.find_all('h2', class_='entry-title')

    for element in article_links:
        title = element.find('a').text
        link = element.find('a')['href']

        # Get the content of the article
        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find('div', class_='entry-content').text.strip()
        content = content.replace('\n', ' ')

        # Add the title and content to the list of articles
        blog_articles.append({
            'title': title,
            'content': content
        })
        
    # Note: The url of Inshorts may have changed since this function was written, 
    # so the BeautifulSoup selectors may need to be updated.

    return blog_articles

# ------------------------------------------------------------------------------------
def get_news_articles():
    """
    Scrape news articles from Inshorts.

    This function scrapes news articles from the 'business', 'sports', 'technology', 
    and 'entertainment' categories on Inshorts. It returns a list of dictionaries, 
    where each dictionary represents one article and has keys for 'title', 'content', 
    and 'category'.

    Returns:
    articles (list of dict): A list of dictionaries, where each dictionary represents 
                             one article and has keys for 'title', 'content', and 'category'.
    """
    # Define the categories to scrape
    categories = ['business', 'sports', 'technology', 'entertainment']
    # Define the base URL
    base_url = 'https://inshorts.com/en/read/'

    # Initialize an empty list to store the articles
    news_articles = []

    # Loop over each category
    for category in categories:
        # Construct the URL of the category page
        category_url = base_url + category

        # Send a GET request to the category page
        response = requests.get(category_url)
        # Parse the response content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all news cards on the page
        news_cards = soup.find_all('div', class_='news-card')

        # Loop over each news card
        for card in news_cards:
            # Find the title and content of the news card
            title = card.find('span', itemprop='headline').text
            content = card.find('div', itemprop='articleBody').text

            # Add the title, content, and category to the list of articles
            news_articles.append({
                'category': category,
                'title': title,
                'content': content
                
            })

    # Note: The url of Inshorts may have changed since this function was written, 
    # so the BeautifulSoup selectors may need to be updated.

    # Return the list of articles (dictionary)
    return news_articles