import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, classification_report
from transformers import Trainer, TrainingArguments
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

# Function to tokenize text data
def tokenize_texts(texts, tokenizer, max_length):
    tokenized = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokenized

# Example data for demonstration
texts = ["I love coding!", "I hate bugs!"] * 1000
labels = [1, 0] * 1000 # Example labels: 1 for positive sentiment, 0 for negative sentiment

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize texts
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 20  # Define maximum length of input
train_tokenized = tokenize_texts(train_texts, tokenizer, max_length)
test_tokenized = tokenize_texts(test_texts, tokenizer, max_length)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(train_tokenized['input_ids'], train_tokenized['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_tokenized['input_ids'], test_tokenized['attention_mask'], torch.tensor(test_labels))

batch_size = 2  # Define batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model
num_classes = 2  # Number of classes (positive and negative sentiment)
model = RobertaClassifier(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 3  # Number of training epochs
for epoch in range(epochs):
    model.train()
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        logits = model(input_ids, attention_mask)
        _, predicted = torch.max(logits, 1)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)
exit()

# Load the dataset
df = pd.read_csv('Grammar Correction.csv', sep=',')

# Preprocess the data
df = df.drop_duplicates()
df['Ungrammatical Statement'] = df['Ungrammatical Statement'].str.strip()
df['Standard English'] = df['Standard English'].str.strip()
df['Ungrammatical Statement'] = df['Ungrammatical Statement'].str.replace(r'^\d+\.\s+', '', regex=True)
df['Standard English'] = df['Standard English'].str.replace(r'^\d+\.\s+', '', regex=True)

# Hardcoded dictionary mapping error types to numerical indices
error_type_to_index = {
    'Sentence Structure Errors': 1,
    'Verb Tense Errors': 2,
    'Subject-Verb Agreement': 3,
    'Article Usage': 4,
    'Spelling Mistakes': 5,
    'Preposition Usage': 6,
    'Punctuation Errors': 7,
    'Relative Clause Errors': 8,
    'Gerund and Participle Errors': 9,
    'Abbreviation Errors': 10,
    'Slang, Jargon, and Colloquialisms': 11,
    'Negation Errors': 12,
    'Incorrect Auxiliaries': 13,
    'Ambiguity': 14,
    'Tautology': 15,
    'Lack of Parallelism in Lists or Series': 16,
    'Mixed Metaphors/Idioms': 17,
    'Parallelism Errors': 18,
    'Contractions Errors': 19,
    'Conjunction Misuse': 20,
    'Inappropriate Register': 21,
    'Passive Voice Overuse': 22,
    'Mixed Conditionals': 23,
    'Faulty Comparisons': 24,
    'Agreement in Comparative and Superlative Forms': 25,
    'Ellipsis Errors': 26,
    'Infinitive Errors': 27,
    'Quantifier Errors': 28,
    'Clichés': 29,
    'Pronoun Errors': 30,
    'Modifiers Misplacement': 31,
    'Run-on Sentences': 32,
    'Word Choice/Usage': 33,
    'Sentence Fragments': 34,
    'Capitalization Errors': 35,
    'Redundancy/Repetition': 36
}

# Map error types to numerical indices
df['label'] = df['Error Type'].map(error_type_to_index)

# Convert DataFrame columns to lists
statements = df['Ungrammatical Statement'].tolist()
labels = df['label'].tolist()

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(statements, labels, test_size=0.2, random_state=42)

# Load the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Tokenize and encode the data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

# Convert the data to PyTorch tensors
print(train_labels)
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                               torch.tensor(train_encodings['attention_mask']),
                                               torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']),
                                              torch.tensor(test_encodings['attention_mask']),
                                              torch.tensor(test_labels))

# Create data loaders
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Set the model for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)




import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the Roberta-based model with a linear head for classification
