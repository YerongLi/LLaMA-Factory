import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, classification_report

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

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['Ungrammatical Statement'], df['label'], test_size=0.2, random_state=42)
# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['Ungrammatical Statement'], df['label'], test_size=0.2, random_state=42)

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

# Train the model
from tqdm import tqdm
import torch.nn.functional as F

epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}'):
        input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch {epoch + 1} Train Loss: {train_loss / len(train_loader)}')

# Evaluate the model
model.eval()
test_preds = []
test_labels_list = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels_list.extend(labels.cpu().numpy())

accuracy = accuracy_score(test_labels_list, test_preds)
print(f'Accuracy: {accuracy}')
print(classification_report(test_labels_list, test_preds))
