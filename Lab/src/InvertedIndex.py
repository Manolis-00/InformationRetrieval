import json
import logging
import math
import pickle
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvertedIndex:
    def __init__(self):
        """
        Initialize the inverted index with necessary data structures.
        """

        # Main inverted index: term -> {doc_id -> [positions]}
        self.index = defaultdict(lambda: defaultdict(list))
        # Document mapping: doc_id -> document metadata
        self.documents = {}
        # Document lengths for normalization
        self.doc_lengths = {}
        # Collection statistics
        self.total_docs = 0
        self.vocabulary_size = 0

    def add_document(self, doc_id, processed_tokens, document_metadata):
        """
        Add a document to the index.

        :param doc_id:
        :param processed_tokens:
        :param document_metadata:
        :return:
        """
        self.documents[doc_id] = document_metadata

        # Calculate document length (number of terms)
        self.doc_lengths[doc_id] = len(processed_tokens)

        # Add terms to index with positions
        for position, term in enumerate(processed_tokens):
            self.index[term][doc_id].append(position)

        self.total_docs += 1

    def get_document_frequency(self, term):
        """
        Get number of documents containing the term.

        :param term:
        :return:
        """
        return len(self.index[term])

    def get_tf_idf(self, term):
        """
        Calculate TF-IDF score for term in document.

        :param term:
        :param doc_id:
        :return:
        """
        if term not in self.index:
            return 0
        return math.log(self.total_docs / self.get_document_frequency(term))

    def save_index(self, file_path):
        """
        Save index to file

        :param file_path:
        :return:
        """
        index_data = {
            'index': dict(self.index),
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'total_docs': self.total_docs,
            'vocabulary_size': len(self.index)
        }

        with open(file_path, 'wb') as f:
            pickle.dump(index_data, f)

    def load_index(self, file_path):
        """
        Load index from file.

        :param file_path:
        :return:
        """
        with open(file_path, 'rb') as f:
            index_data = pickle.load(f)

        self.index = defaultdict(lambda: defaultdict(list), index_data['index'])
        self.documents = index_data['documents']
        self.doc_lengths = index_data['doc_lengths']
        self.total_docs = index_data['total_docs']
        self.vocabulary_size = index_data['vocabulary_size']

    def get_statistics(self):
        """
        Get index statistics

        :param self:
        :return:
        """
        return {
            'total_documents': self.total_docs,
            'vocabulary_size': len(self.index),
            'average_document_length': sum(self.doc_lengths.values()) / max(1, self.total_docs),
            'total_terms': sum(
                len(positions)
                for doc_dict in self.index.values()
                for positions in doc_dict.values()
            )
        }


def build_inverted_index(processed_articles_file, index_output_file):
    """
    Build inverted index from processed articles.

    :param processed_articles_file:
    :param index_output_file:
    :return:
    """
    try:
        # Load processed articles
        logger.info(f"Loading processed articles from {processed_articles_file}")
        with open(processed_articles_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        # Create inverted index
        index = InvertedIndex()

        # Add documents to index
        for i, article in enumerate(articles, 1):
            try:
                # Use URL as document ID (assuming it's unique)
                doc_id = article['url']

                # Create document metadata
                metadata = {
                    'title': article['title'],
                    'url': article['url'],
                    'timestamp': article['timestamp']
                }

                # Add to index
                index.add_document(
                    doc_id=doc_id,
                    processed_tokens=article['processed_content'],
                    document_metadata=metadata
                )

                if i % 10 == 0:
                    logger.info(f"Indexed {i} documents")

            except KeyError as e:
                logger.error(f"Missing required field in article {i}: {e}")
                continue

        # Save index
        logger.info(f"Saving index to {index_output_file}")
        index.save_index(index_output_file)

        # Get and return statistics
        stats = index.get_statistics()
        logger.info("\nIndex Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        return stats

    except Exception as e:
        logger.error(f"Error building index: {e}")
        return None


if __name__ == "__main__":
    lemmatized_file = "lemmatized_information retrieval_articles.json"
    stemmed_file = "stemmed_information retrieval_articles.json"
    index_lemmatized_file = "inverted_lemmatized_index.pkl"
    index_stemmed_file = "inverted_stemmed_index.pkl"

    lemmatized_stats = build_inverted_index(lemmatized_file, index_lemmatized_file)
    stemmed_stats = build_inverted_index(stemmed_file, index_stemmed_file)

    lemmatized_index = InvertedIndex()
    lemmatized_index.load_index(index_lemmatized_file)

    stemmed_index = InvertedIndex()
    stemmed_index.load_index(index_stemmed_file)

    lemmatized_score = lemmatized_index.get_tf_idf("run")
    stemmed_score = stemmed_index.get_tf_idf("run")
