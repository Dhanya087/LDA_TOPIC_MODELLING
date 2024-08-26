from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the 20 Newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', categories=['rec.autos', 'sci.med'], remove=('headers', 'footers', 'quotes'))

# Preprocess the documents
documents = newsgroups_data.data[:10]  # Select a subset of documents for this example

# Step 2: Create LDA-Based Vectors

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
doc_term_matrix = vectorizer.fit_transform(documents)

# Fit the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_vectors = lda.fit_transform(doc_term_matrix)

# Step 3: Create Word Embeddings using Pre-trained Model

# Load pre-trained word vectors (Word2Vec model from gensim's model zoo)
word2vec_model = api.load("word2vec-google-news-300")

# Tokenize the documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Create document embeddings by averaging pre-trained word vectors
word_embeddings = np.array([np.mean([word2vec_model[word] for word in doc if word in word2vec_model]
                                    or [np.zeros(300)], axis=0) for doc in tokenized_docs])

# Step 4: Compute Cosine Similarity for Both Approaches

# LDA-Based Cosine Similarity
lda_cosine_sim = cosine_similarity(lda_vectors)

# Word Embedding-Based Cosine Similarity
word_embedding_cosine_sim = cosine_similarity(word_embeddings)

# Step 5: Cluster the Documents Based on Similarity Scores

# Clustering LDA vectors
kmeans_lda = KMeans(n_clusters=2, random_state=42).fit(lda_vectors)
lda_clusters = kmeans_lda.labels_

# Clustering Word Embedding vectors
kmeans_we = KMeans(n_clusters=2, random_state=42).fit(word_embeddings)
we_clusters = kmeans_we.labels_

# Step 6: Visualize the Clusters

# Visualize LDA clusters using t-SNE
tsne_lda = TSNE(n_components=2, random_state=42, perplexity=5)  # Set perplexity to less than the number of samples
lda_tsne = tsne_lda.fit_transform(lda_vectors)
plt.figure(figsize=(8, 6))
plt.scatter(lda_tsne[:, 0], lda_tsne[:, 1], c=lda_clusters, cmap='viridis')
plt.title("LDA-Based Document Clustering")
plt.show()

# Visualize Word Embedding clusters using t-SNE
tsne_we = TSNE(n_components=2, random_state=42, perplexity=5)  # Set perplexity to less than the number of samples
we_tsne = tsne_we.fit_transform(word_embeddings)
plt.figure(figsize=(8, 6))
plt.scatter(we_tsne[:, 0], we_tsne[:, 1], c=we_clusters, cmap='plasma')
plt.title("Word Embedding-Based Document Clustering")
plt.show()

# Step 7: Analysis
print("LDA-Based Cosine Similarity:")
print(lda_cosine_sim)

print("\nWord Embedding-Based Cosine Similarity:")
print(word_embedding_cosine_sim)

print("\nLDA-Based Clusters:", lda_clusters)
print("Word Embedding-Based Clusters:", we_clusters)
