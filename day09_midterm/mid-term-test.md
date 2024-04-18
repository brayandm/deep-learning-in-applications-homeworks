# Deep Learning in Applications Mid-Term Test

## Name: Brayan Duran Medina

### 1. Word representations: basic approaches (BoW, TF-IDF).

Word representations are techniques used to mathematically map words to vectors of real numbers which helps in processing natural language data and performing machine learning tasks that involve textual input.

The Bag of Words (BoW) is one of the simplest methods for getting numerical features from text. This method transforms the text into a "bag" of words, neglecting grammar and the order of words, focusing only on the occurrence of words in the document.

Term Frequency-Inverse Document Frequency (TF-IDF) improves on the basic idea of BoW by considering not just the frequency of words in a single document but in the entire corpus of documents. It helps identify which words are most relevant to a document within a larger context.

### 2. Word embeddings (word2vec: key ideas, linearity, skip-gram, negative sampling).

Word embeddings are a class of techniques where individual words are represented as real-valued vectors in a predefined vector space. Word2Vec is one of the most popular techniques to create word embedding models. The key idea behind Word2Vec is to learn word embeddings by predicting the context of a word given its neighboring words. The linearity of word embeddings means that the relationship between words can be represented as linear transformations in the vector space. The skip-gram model is a variant of Word2Vec that predicts the context words given a target word. Negative sampling is a training method used in Word2Vec to simplify the learning objective by teaching the model to distinguish a target word from a few negative examples (noise words), rather than predicting the exact word from the entire vocabulary.

### 3. Seq2seq model. Beam search.

A sequence to sequence model is a type of deep neural network designed to convert sequences from one domain to another, such as translating sentences from English to French. It typically uses recurrent neural networks to process the input sequence (encoder) and generate the output sequence (decoder), one element at a time. This model is fundamental in tasks that require transformations of sequential data, including machine translation, speech recognition, and text summarization.

Beam search is a heuristic search algorithm used in sequence prediction problems, particularly with models like sequence to sequence that generate sequences of data (e.g., text translation). Unlike greedy search which selects the single best option at each step, beam search considers multiple alternatives based on a specified beam width, keeping the top 'k' most likely sequences at each step of the model's output. This approach balances between finding a sufficiently good solution efficiently and exploring a broad set of potential solutions, making it more likely to find a better output sequence than methods considering only one path.

### 4. Machine translation metrics.

Machine translation metrics are used to evaluate the quality of translation produced by machine translation systems relative to human translations. The most common metrics include:

BLEU (Bilingual Evaluation Understudy): Measures the correspondence of n-grams between the machine's output and reference translations, providing a score from 0 to 100; higher scores indicate better translations, emphasizing precision.

METEOR (Metric for Evaluation of Translation with Explicit ORdering): Expands on BLEU by considering synonyms and stemming, and by balancing precision and recall, providing a more exact assessment.

TER (Translation Edit Rate): Calculates the number of edits (insertions, deletions, substitutions) required to change a machine translation output into one of the references, offering insights into the translation's accuracy.

### 5. Attention and self-attention mechanisms.

Attention mechanisms in neural networks are techniques that mimic cognitive attention by focusing on specific parts of the input sequentially when producing the output, enhancing the model's ability to learn and interpret dependencies in data. These mechanisms are particularly useful in sequence-to-sequence tasks, such as language translation, where they help the model to weigh the importance of different input parts differently, improving the context awareness of the output.

Self-attention, a specific form of attention mechanism, allows inputs to interact with each other and identify dependencies regardless of their positions in the input sequence. This is achieved through a set of learned weights that help the network emphasize the most relevant parts of the input data to better process the entire sequence. Self-attention is a key component of Transformer models, which have set new standards in performance across a range of language processing tasks by enabling much deeper interpretation of sequences in a highly efficient manner.

### 8. BERT: main ideas (masked language modeling, pre-training on two tasks).

BERT (Bidirectional Encoder Representations from Transformers) is a groundbreaking model in NLP developed by Google, primarily known for its use of masked language modeling (MLM) and its novel pre-training approach. The main ideas behind BERT are:

Masked Language Modeling (MLM): Unlike traditional language models that predict the next word in a sequence, BERT randomly masks words in the input sequences and predicts these masked words. This allows the model to learn a deep bidirectional representation of the input sequence, considering both left and right context in all layers of the model.

Pre-training on Two Tasks: BERT is pre-trained using two tasks: the aforementioned MLM and Next Sentence Prediction (NSP). In NSP, the model learns to predict whether one sentence logically follows another, which helps in understanding relationship between sentences. This pre-training on large corpora enables BERT to capture a rich language representation, making it highly effective when fine-tuned for specific tasks like question answering, sentiment analysis, and more.

### 10. Compare WordPiece and BPE tokenizers.

WordPiece and Byte Pair Encoding (BPE) are subword tokenization methods essential for enhancing the performance of neural models like BERT and GPT, particularly in handling out-of-vocabulary words across diverse languages. BPE begins by merging the most frequent adjacent character pairs into subwords and continues this process until a predetermined vocabulary size is reached, making it effective for models such as OpenAI's GPT. In contrast, WordPiece extends upon BPE’s methodology by choosing merges that maximize the likelihood of the text under a language model, which improves the handling of morphologically complex languages, a feature extensively utilized by Google in systems like BERT.
