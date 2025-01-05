import pickle
from RankingModels import RankingModels
from TextPreprocessor import TextPreprocessor

class SearchEngine:
    def __init__(self, index_path):
        """Initialize search engine with index and ranking models"""
        self.index_data = self.load_index(index_path)
        self.ranking_models = RankingModels(self.index_data)
        self.preprocessor = TextPreprocessor(use_stemming=False)  # Match the index preprocessing

    def load_index(self, file_path):
        """Load the inverted index from disk"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def preprocess_query(self, query):
        """Preprocess the query using the same pipeline as documents"""
        return self.preprocessor.process_text(query)

    def search(self, query, ranking_method='bm25', max_results=10):
        """
        Search documents using the specified ranking method.

        Args:
            query: User's search query
            ranking_method: Method to rank results ('tfidf', 'vsm', or 'bm25')
            max_results: Maximum number of results to return

        Returns:
            List of tuples (document metadata, score)
        """
        # Preprocess query
        query_tokens = self.preprocess_query(query)

        # Get ranked documents
        ranked_docs = self.ranking_models.rank_documents(query_tokens, ranking_method)

        # Format results with document metadata
        results = []
        for doc_id, score in ranked_docs[:max_results]:
            doc_metadata = self.index_data['documents'][doc_id]
            results.append((doc_metadata, score))

        return results

def display_results(results):
    """Display search results in a formatted way"""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} results:")
    print("-" * 80)

    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. {doc['title']}")
        print(f"   Score: {score:.4f}")
        print(f"   URL: {doc['url']}")
        print("-" * 80)

def main():
    """Main function to run the search engine"""
    # Initialize search engine with the lemmatized index
    search_engine = SearchEngine('inverted_lemmatized_index.pkl')

    ranking_methods = {
        '1': 'tfidf',
        '2': 'vsm',
        '3': 'bm25'
    }

    while True:
        print("\nSearch Engine")
        print("------------")
        print("Ranking Methods:")
        print("1. TF-IDF")
        print("2. Vector Space Model")
        print("3. Okapi BM25")
        print("\nEnter 'exit' to quit")

        # Get ranking method
        method_choice = input("\nSelect ranking method (1-3): ")
        if method_choice.lower() == 'exit':
            break

        if method_choice not in ranking_methods:
            print("Invalid choice. Please select 1-3.")
            continue

        # Get search query
        query = input("Enter your search query: ")
        if query.lower() == 'exit':
            break

        # Perform search and display results
        results = search_engine.search(
            query,
            ranking_method=ranking_methods[method_choice],
            max_results=10
        )
        display_results(results)

if __name__ == "__main__":
    main()