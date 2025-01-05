import json
import logging
import re
from venv import logger

import nltk
nltk.download('punkt_tab')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Configure logging to track processing progress and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, use_stemming=True):
        """
        A class for preprocessing text data using NLP techniques.
        Supports both stemming and lemmatization for word normalization.
        """

        # Download required NLTK resources for text processing
        try:
            nltk.download('punkt', quiet=True)                          # For tokenization
            nltk.download('stopwords', quiet=True)                      # For stopword removal
            nltk.download('wordnet', quiet=True)                        # For lemmatization
            nltk.download('averaged_perceptron_tagger', quiet=True)     # For POS tagging
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
            raise

        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        # Choose between stemming and lemmatization based on initialization parameter
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if not use_stemming else None

    def clean_text(self, text):
        """
        Clean the input text by removing special characters and standardizing format

        Args:
            text (str): Input text to be cleaned

        Returns:
            str: Cleaned text in lowercase with only alphabetic characters and spaces
        """
        if not isinstance(text, str):
            return ""

        # Remove all characters except letters and spaces using regex
        text = re.sub(r'[^a-zA-z\s]', '', text)

        # Standardize text by converting to lowercase
        return text.lower()

    def process_text(self, text):
        """
        Apply full text processing pipeline: cleaning, tokenization, stopword removal,
        and word normalization (stemming or lemmatization)

        Args:
            text (str): Raw input text

        Returns:
            list: Processed tokens after applying the complete pipeline
        """

        # Step 1: Clean the text
        cleaned_text = self.clean_text(text)

        # Step 2: Split text into individual tokens
        tokens = word_tokenize(cleaned_text)

        # Step 3: Remove common stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        # Step 4: Apply word normalization
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        else:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens


def process_wikipedia_data(input_filename, output_filename, use_stemming=True):
    """
    Process Wikipedia articles from JSON file using the text preprocessing pipeline

    Args:
        input_filename (str): Path to input JSON file containing Wikipedia articles
        output_filename (str): Path where processed articles will be saved
        use_stemming (bool): Whether to use stemming or lemmatization

    Returns:
        dict: Processing statistics or None if processing fails
    """

    try:
        # Read input JSON file
        logger.info(f"Loading articles from {input_filename}")
        with open(input_filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        # Validate input data structure
        if not isinstance(articles, list):
            raise ValueError("Input JSON must contain a list of articles")

        # Create text processor instance
        preprocessor = TextPreprocessor(use_stemming=use_stemming)

        # Process each article
        processed_articles = []
        for i, article in enumerate(articles, 1):
            try:
                # Create processed article object with both original and processed content
                processed_article = {
                    'title': article['title'],
                    'url': article['url'],
                    'timestamp': article['timestamp'],
                    'processed_content': preprocessor.process_text(article['content']),
                    'original_content': article['content']
                }
                processed_articles.append(processed_article)

                # Log progress for every 10 articles
                if i % 10 == 0:
                    logger.info(f"Processed {i} articles")

            except KeyError as e:
                logger.error(f"Missing required field in article {i}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")
                continue

        # Save processed results to output file
        logger.info(f"Saving processed articles to{output_filename}")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, ensure_ascii=False, indent=2)

        # Return processing statistics
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
    # Configuration for processing Wikipedia articles
    input_file = "wikipedia_information retrieval_articles.json"
    use_stemming = False
    if use_stemming:
        output_file = "stemmed_information retrieval_articles.json"
    else:
        output_file = "lemmatized_information retrieval_articles.json"

    # Process articles and display statistics
    stats = process_wikipedia_data(input_file, output_file, use_stemming)

    if stats:
        logger.info("\nProcessing Statistics:")
        logger.info(f"Input articles: {stats['total_input_articles']}")
        logger.info(f"Successfully processed: {stats['total_processed_articles']}")
        logger.info(f"Processing method: {stats['processing_method']}")
