import json
import time
from venv import logger

import requests
from bs4 import BeautifulSoup, NavigableString

def get_wikipedia_articles(search_term, max_articles=100):
    """
    Crawl Wikipedia articles related to the search term.

    :param search_term (str): Term to search for
    :param max_articles (int): Maximum number of articles to collect

    :return: List of dictionaries containing article data
    """

    articles = []
    base_url = "https://en.wikipedia.org/w/api.php"

    # Parameters for the Wikipedia API
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": search_term,
        "srlimit": max_articles
    }

    try:
        #Get search results
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        search_results = response.json()

        # Process each search result
        for result in search_results["query"]["search"]:
            #article_url = f"https://en.wikipedia.org/wiki/{result['title'].replace(' ', '_')}"

            # Encode title for url
            encoded_title = requests.utils.quote(result['title'])
            article_url = f"https://en.wikipedia.org/wiki/{encoded_title}"

            try:
                # Get full article content
                article_response = requests.get(article_url)
                article_response.raise_for_status()

                soup = BeautifulSoup(article_response.text, 'html.parser')

                content_strategies = [
                    lambda: soup.find(id="mw-content-text"), # Original method
                    lambda: soup.find("div", class_="mw-parser-output"), # Alternative class
                    lambda: soup.find("div", id="content"), # Another possible div
                ]


                content_div = None
                for strategy in content_strategies:
                    content_div = strategy()
                    if content_div:
                        break

                if not content_div:
                    logger.warning(f"Could not find content for article: {result['title']}")
                    continue

                # Remove unwanted elements
                if content_div.find(["table", "sup", "span.mw-editsection", "div.hatnote", "div.metadata"]):
                    for unwanted in content_div.find(["table", "sup", "span.mw-editsection", "div.hatnote", "div.metadata"]):
                        if not isinstance(unwanted, NavigableString):
                            unwanted.decompose()

                # Get clean text
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text(strip=True)])

                # Only add articles with meaningful content
                if len(content) > 100:
                    article_data = {
                        "title": result["title"],
                        "url": article_url,
                        "content": content,
                        "timestamp": result.get("timestamp", "")
                    }

                    articles.append(article_data)
                    logger.info(f"Collected article: {result['title']}")

                time.sleep(1)

            except requests.RequestException as article_error:
                logger.error(f"Error fetching article {result['title']}: {article_error}")

            """
                        # Extract main content
            content_div = soup.find(id="mw-context-text")
            if content_div:
                # Remove unwanted elements
                for unwanted in content_div.find_all(["table", "sup", "span.mw-editsection"]):
                    unwanted.decompose()

                # Get clean text
                content = ' '.join([p.get_text().strip() for p in content_div.find_all('p')])

                article_data = {
                    "title": result["title"],
                    "url": article_url,
                    "content": content,
                    "timestamp": result["timestamp"]
                }

                articles.append(article_data)

                time.sleep(1)

            """

    except requests.RequestException as exception:
        print(f"Error occured while fetching data: {exception}")

    return articles

def save_data(articles, base_filename):
    """
    Save articles in both CSV and JSON Formats

    :param articles:
    :param base_filename:
    :return:
    """

    # Save as JSON
    json_filename = f"{base_filename}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    # Save as CSV
    #csv_filename = f"{base_filename}.csv"


if __name__ == "__main__":
    search_term = "running"
    articles = get_wikipedia_articles(search_term, max_articles=5000)

    if articles:
        #json_file, csv_file = save_data(articles, f"wikipedia_{search_term}_articles")
        json_file = save_data(articles, f"wikipedia_{search_term}_articles")
        logger.info(f"\nData collection complete!")
        logger.info(f"Number of articles collected: {len(articles)}")
        logger.info(f"Files saved:")
        logger.info(f"JSON: {json_file}")