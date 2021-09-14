import pandas as pd
import spacy
from spacy.util import minibatch
import random

# Loading the spam data
# ham is the label for non-spam messages
spam = pd.read_csv('sample.csv')
spam.head(10)

# Create an empty model
nlp = spacy.blank("en")

# Create the TextCategorizer with exclusive classes and "bow" architecture
textcat = nlp.create_pipe(
    "textcat",
    config={
        "exclusive_classes": True,
        "architecture": "bow"})

# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)

# Add labels to text classifier
textcat.add_label("script")
textcat.add_label("defect")

train_texts = spam['text'].values
train_labels = [{'cats': {'script': label == 'script',
                          'defect': label == 'defect'}}
                for label in spam['label']]
train_data = list(zip(train_texts, train_labels))
train_data[:3]

spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

# Create the batch generator with batch size = 8
batches = minibatch(train_data, size=8)
# Iterate through minibatches
for batch in batches:
    # Each batch is a list of (text, label) but we need to
    # send separate lists for texts and labels to update().
    # This is a quick way to split a list of tuples into lists
    texts, labels = zip(*batch)
    nlp.update(texts, labels, sgd=optimizer)

random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        # Each batch is a list of (text, label) but we need to
        # send separate lists for texts and labels to update().
        # This is a quick way to split a list of tuples into lists
        texts, labels = zip(*batch)
        nlp.update(texts, labels, sgd=optimizer, losses=losses)
    # print(losses)

texts = ["File mot found",
         "exception not found",
         "button issue",
         "click not working"]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

# print(scores)

# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])