import pandas as pd
import spacy
from spacy.util import minibatch
import random
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

def cosine_similarity(a, b):
    return a.dot(b)/np.sqrt(a.dot(a) * b.dot(b))

# Loading the spam data
# ham is the label for non-spam messages
spam = pd.read_csv('sample.csv')
spam.head(10)

# Create an large model
nlp = spacy.load("en_core_web_sm")

# Create the TextCategorizer with exclusive classes and "bow" architecture
with nlp.disable_pipes():
    doc_vectors = np.array([nlp(text).vector for text in spam.text])

doc_vectors.shape


X_train, X_test, y_train, y_test = train_test_split(doc_vectors, spam.label,
                                                    test_size=0.1, random_state=1)

# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)
print(f"Accuracy: {svc.score(X_test, y_test) * 100:.3f}%", )

a = nlp("Test error button").vector
b = nlp("Exception not found").vector
print(cosine_similarity(a, b))



