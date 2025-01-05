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
        Initialize the inverted index with core data structures:
        - index: A nested defaultdict storing term -> document -> positions mapping
        - documents: Dictionary storing document metadata
        - doc_lengths: Dictionary storing document lengths for normalization
        - Collection statistics for TF-IDF calculations
        """

        # Main inverted index: term -> {doc_id -> [positions]}
        self.index = defaultdict(lambda: defaultdict(list))
        # Store document metadata separately for efficient access
        self.documents = {}
        # Track document lengths for scoring normalization
        self.doc_lengths = {}
        # Collection-level statistics
        self.total_docs = 0
        self.vocabulary_size = 0

    def add_document(self, doc_id, processed_tokens, document_metadata):
        """
        Add a document to the index while maintaining position information.
        Updates document statistics and term frequencies atomically.

        Args:
            doc_id: Unique identifier for the document
            processed_tokens: List of preprocessed terms from the document
            document_metadata: Additional document information (title, URL, etc.)
        """

        # Store document metadata for quick access
        self.documents[doc_id] = document_metadata

        # Record document length for normalization in scoring
        self.doc_lengths[doc_id] = len(processed_tokens)

        # Index each term with its position for phrase queries
        for position, term in enumerate(processed_tokens):
            self.index[term][doc_id].append(position)

        # Index each term with its position for phrase queries
        self.total_docs += 1

    def get_document_frequency(self, term):
        """
        Calculate the document frequency (number of documents containing the term).

        Args:
            term: The term to look up

        Returns:
            Number of documents containing the term
        """
        return len(self.index[term])

    def get_tf_idf(self, term):
        """
        Calculate the IDF (Inverse Document Frequency) score for a term.
        Uses logarithmic scaling to dampen the effect of large differences.

        Args:
            term: The term to calculate IDF for

        Returns:
            IDF score or 0 if term not in index
        """
        if term not in self.index:
            return 0
        return math.log(self.total_docs / self.get_document_frequency(term))

    def save_index(self, file_path):
        """
        Serialize and save the index to disk.
        Converts defaultdict to regular dict for pickling.

        Args:
            file_path: Path to save the index
        """

        # Prepare index data for serialization
        index_data = {
            'index': dict(self.index),
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'total_docs': self.total_docs,
            'vocabulary_size': len(self.index)
        }

        # Save using pickle for efficient binary serialization
        with open(file_path, 'wb') as f:
            pickle.dump(index_data, f)

    def load_index(self, file_path):
        """
        Load a previously saved index from disk.
        Reconstructs the defaultdict structure.

        Args:
            file_path: Path to the saved index
        """

        # Load serialized index data
        with open(file_path, 'rb') as f:
            index_data = pickle.load(f)

        # Reconstruct index structures
        self.index = defaultdict(lambda: defaultdict(list), index_data['index'])
        self.documents = index_data['documents']
        self.doc_lengths = index_data['doc_lengths']
        self.total_docs = index_data['total_docs']
        self.vocabulary_size = index_data['vocabulary_size']

    def get_statistics(self):
        """
        Calculate and return key statistics about the index.

        Returns:
            Dictionary containing index statistics
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
    Build an inverted index from preprocessed articles.
    Handles both lemmatized and stemmed versions of documents.

    Args:
        processed_articles_file: Path to preprocessed articles JSON
        index_output_file: Path to save the built index

    Returns:
        Dictionary of index statistics or None on error
    """
    try:
        # Load preprocessed document collection
        logger.info(f"Loading processed articles from {processed_articles_file}")
        with open(processed_articles_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        # Initialize new index
        index = InvertedIndex()

        # Process documents incrementally
        for i, article in enumerate(articles, 1):
            try:
                # Use URL as unique document identifier
                doc_id = article['url']

                # Extract and store document metadata
                metadata = {
                    'title': article['title'],
                    'url': article['url'],
                    'timestamp': article['timestamp']
                }

                # Add document to index
                index.add_document(
                    doc_id=doc_id,
                    processed_tokens=article['processed_content'],
                    document_metadata=metadata
                )

                # Log progress for every 10 documents
                if i % 10 == 0:
                    logger.info(f"Indexed {i} documents")

            except KeyError as e:
                logger.error(f"Missing required field in article {i}: {e}")
                continue

        # Save completed index
        logger.info(f"Saving index to {index_output_file}")
        index.save_index(index_output_file)

        # Calculate and log final statistics
        stats = index.get_statistics()
        logger.info("\nIndex Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        return stats

    except Exception as e:
        logger.error(f"Error building index: {e}")
        return None


    #συναρτηση εκτυπωσεις inverted index
def print_inverted_index(index):
    """
    Εμφανίζει το αντιστραμμένο ευρετήριο με τις λέξεις και τις θέσεις τους στα έγγραφα.
    """
    for term, docs in index.items():
        print(f"Term: {term}")
        for doc_id, positions in docs.items():
            print(f"  Document: {doc_id} -> Positions: {positions}")
        print("-" * 50)  # Διαχωριστικό γραμμής για καλύτερη ευκρίνεια


if __name__ == "__main__":
    # File paths for different preprocessing approaches
    lemmatized_file = "lemmatized_information retrieval_articles.json"
    stemmed_file = "stemmed_information retrieval_articles.json"
    index_lemmatized_file = "inverted_lemmatized_index.pkl"
    index_stemmed_file = "inverted_stemmed_index.pkl"

    # Build indices for both preprocessing approaches
    lemmatized_stats = build_inverted_index(lemmatized_file, index_lemmatized_file)
    stemmed_stats = build_inverted_index(stemmed_file, index_stemmed_file)

    # Load indices for comparison
    lemmatized_index = InvertedIndex()
    lemmatized_index.load_index(index_lemmatized_file)

    stemmed_index = InvertedIndex()
    stemmed_index.load_index(index_stemmed_file)

    # Compare TF-IDF scores for example term
    lemmatized_score = lemmatized_index.get_tf_idf("run")
    stemmed_score = stemmed_index.get_tf_idf("run")

    print("Inverted Index for Lemmatized Documents:")
    print_inverted_index(lemmatized_index.index)

    print("\nInverted Index for Stemmed Documents:")
    print_inverted_index(stemmed_index.index)