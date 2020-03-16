- Run `create_stemmed_wordlist_vectors.py` with location for conceptnet-numberbatch word embeddings to process the wordvectors, stem the word associated with the vector and generates the final wordlist. 
- Run `question_preprocessor.py` with location for question pair dataset, to preprocess questions by cleaning and generating sentence vectors.
- Run`lstm.py` to train the LSTM siamese model with batching function.
- Model can be directly trained without regenerating question vectors.
- Download the Quora question pair dataset from [here](https://drive.google.com/open?id=188CJNYj8c4cSjq2ZQFCf8YqhDH_FWhlh)
- Download the ConceptNet Numberbatch pretrained embedding from [here](https://github.com/commonsense/conceptnet-numberbatch)
- Libraries required are tensorflow, nltk, pandas, numpy