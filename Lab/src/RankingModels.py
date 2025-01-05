import math
from collections import defaultdict, Counter


class RankingModels:
    def __init__(self, index_data):
        """
        Initialize ranking models with index data.

        :param index_data: Dictionary containing inverted index and documented statistics
        """
        self.index = index_data['index']
        self.doc_lengths = index_data['doc_lengths']
        self.total_docs = index_data['total_docs']
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(1, self.total_docs)


    def calculate_tf_idf(self, query_tokens):
        """
        Calculate TF-IDF scores for documents matching the query.

        :param query_tokens: List of preprocessed query terms

        :return: Dictionary of document IDs and their TF-IDF scores
        """
        scores = defaultdict(float)

        for term in query_tokens:
            if term in self.index:
                # Calculate IDF
                idf = math.log(self.total_docs / len(self.index[term]))

                # Calculate TF-IDF for each matching documents
                for doc_id, positions in self.index[term].items():
                    tf = len(positions)
                    scores[doc_id] += tf * idf

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


    def vector_space_model(self, query_tokens):
        """
        Implement Vector Space Model with cosine similarity.

        :param query_tokens: List of preprocessed query terms.
        :return: Dictionary of document IDs and their cosine similarity scores
        """

        # Calculate query term frequencies
        query_tf = Counter(query_tokens)

        # Calculate document scores using cosine similarity
        scores = defaultdict(float)
        query_magnitude = math.sqrt(sum(freq ** 2 for freq in query_tf.values()))

        for term, query_term_freq in query_tf.items():
            if term in self.index:
                idf = math.log(self.total_docs / len(self.index[term]))
                query_weight = query_term_freq * idf

                for doc_id, positions in self.index[term].items():
                    doc_tf = len(positions)
                    doc_weight = doc_tf * idf

                    # Accumulate dot product
                    scores[doc_id] += query_weight * doc_weight

        # Normalize scores by document magnitude
        for doc_id in scores:
            doc_magnitude = math.sqrt(sum(
                (len(positions) * math.log(self.total_docs / len(self.index[term]))) ** 2
                for term, doc_dict in self.index.items()
                for d_id, positions in doc_dict.items()
                if d_id == doc_id
            ))
            if doc_magnitude > 0:
                scores[doc_id] = scores[doc_id] / (query_magnitude * doc_magnitude)

        return dict(sorted(scores.items(), key=lambda  x: x[1], reverse=True))


    def okapi_bm25(self, query_tokens, k1=1.5, b=0.75):
        """
        Implement Okapi BM25 ranking algorithm.

        :param query_tokens: List of preprocessed query terms
        :param k1: Term frequency saturation parameter (default: 1.5)
        :param b: Length normalization parameter (default: 0.75)
        :return: Dictionary of document IDs and their BM25 scores
        """
        scores = defaultdict(float)

        for term in query_tokens:
            if term in self.index:
                # Calculate IDF component
                n_docs_with_term = len(self.index[term])
                idf = math.log((self.total_docs - n_docs_with_term + 0.5) / (n_docs_with_term + 0.5) + 1)

                for doc_id, positions in self.index[term].items():
                    # Calculate normalized term frequency
                    tf = len(positions)
                    doc_length = self.doc_lengths[doc_id]
                    length_norm = 1 - b + b * (doc_length / self.avg_doc_length)

                    # Calculate BM25 score for term-document pair
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 *length_norm
                    scores[doc_id] += idf * numerator / denominator

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def rank_documents(self, query_tokens, method='bm25'):
        """
        Rank documents using the specified method.

        :param query_tokens: List of preprocessed query tokens
        :param method: Ranking method to use ('tfidf', 'vsm', or 'bsm25')

        :return: List of tuples (doc_id, score) sorted by score
        """
        if method == 'tfidf':
            scores = self.calculate_tf_idf(query_tokens)
        elif method == 'vsm':
            scores = self.vector_space_model(query_tokens)
        elif method == 'bm25':
            scores = self.okapi_bm25(query_tokens)
        else:
            raise ValueError("Invlaid ranking method. Choose 'tfidf', 'vsm', or 'bm25'.")

        return list(scores.items())