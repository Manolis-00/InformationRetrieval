import json
import logging
import re
from venv import logger

import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, use_stemming=True):
        """
        Initialize the text preprocessor with specified options

        :param use_stemming (bool): If True, use stemming; If False use lemmatization
        """

        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
            raise

        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if not use_stemming else None

    def clean_text(self, text):
        """
        Remove special characters and convert to lowercase.

        :param text:
        :return:
        """
        if not isinstance(text, str):
            return ""

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-z\s]', '', text)

        # Convert to lowercase
        return text.lower()

    def process_text(self, text):
        """
       Apply full text processing

       :param self:
       :param text:
       :return:
       """

        # Clean text
        cleaned_text = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(cleaned_text)

        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        # Apply stemming or lemmatization
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens


def process_wikipedia_data(input_filename, output_filename, use_stemming=True):
    """
    Process Wikipedia articles data from the JSON file

    :param input_filename:
    :param output_filename:
    :param use_stemming:
    :return:
    """

    try:
        # Load file
        logger.info(f"Loading articles from {input_filename}")
        with open(input_filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        if not isinstance(articles, list):
            raise ValueError("Input JSON must contain a list of articles")

        # Initialize preprocessor
        preprocessor = TextPreprocessor(use_stemming=use_stemming)

        # Process articles
        processed_articles = []
        for i, article in enumerate(articles, 1):
            try:
                processed_article = {
                    'title': article['title'],
                    'url': article['url'],
                    'timestamp': article['timestamp'],
                    'processed_content': preprocessor.process_text(article['content']),
                    'original_content': article['content']
                }
                processed_articles.append(processed_article)

                if i % 10 == 0:
                    logger.info(f"Processed {i} articles")

            except KeyError as e:
                logger.error(f"Missing required field in article {i}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                continue

        # Save processed artiles
        logger.info(f"Saving processed articles to{output_filename}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, ensure_ascii=False, indent=2)

        # Return statistics
        return {
            'total_input_articles': len(articles),
            'total_processed_articles': len(processed_articles),
            'processing_method': 'stemming' if use_stemming else 'lemmatization'
        }

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filename}")
        return None
    except json.JSONDecoder:
        logger.error(f"Invalid JSON in input file: {input_filename}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


if __name__ == "__main__":
    # Process the wikipedia articles that were collected
    input_file = "wikipedia_information retrieval_articles.json"
    use_stemming = True
    if use_stemming:
        output_file = "stemmed_information retrieval_articles.json"
    else:
        output_file = "lemmatized_information retrieval_articles.json"

    stats = process_wikipedia_data(input_file, output_file, use_stemming)

    if stats:
        logger.info("\nProcessing Statistics:")
        logger.info(f"Input articles: {stats['total_input_articles']}")
        logger.info(f"Successfully processed: {stats['total_processed_articles']}")
        logger.info(f"Processing method: {stats['processing_method']}")
