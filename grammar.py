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
from tqdm import tqdm

error_type_to_index = {
    'Sentence Structure Errors': 0,
    'Verb Tense Errors': 1,
    'Subject-Verb Agreement': 2,
    'Article Usage': 3,
    'Spelling Mistakes': 4,
    'Preposition Usage': 5,
    'Punctuation Errors': 6,
    'Relative Clause Errors': 7,
    'Gerund and Participle Errors': 8,
    'Abbreviation Errors': 9,
    'Slang, Jargon, and Colloquialisms': 10,
    'Negation Errors': 11,
    'Incorrect Auxiliaries': 12,
    'Ambiguity': 13,
    'Tautology': 14,
    'Lack of Parallelism in Lists or Series': 15,
    'Mixed Metaphors/Idioms': 16,
    'Parallelism Errors': 17,
    'Contractions Errors': 18,
    'Conjunction Misuse': 19,
    'Inappropriate Register': 20,
    'Passive Voice Overuse': 21,
    'Mixed Conditionals': 22,
    'Faulty Comparisons': 23,
    'Agreement in Comparative and Superlative Forms': 24,
    'Ellipsis Errors': 25,
    'Infinitive Errors': 26,
    'Quantifier Errors': 27,
    'Clichés': 28,
    'Pronoun Errors': 29,
    'Modifiers Misplacement': 30,
    'Run-on Sentences': 31,
    'Word Choice/Usage': 32,
    'Sentence Fragments': 33,
    'Capitalization Errors': 34,
    'Redundancy/Repetition': 35
}


# Map error types to numerical indices
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example data for demonstration
# texts = ["I love coding!", "I hate bugs!"] * 100
# labels = [1, 0] * 100 # Example labels: 1 for positive sentiment, 0 for negative sentiment
df = pd.read_csv('Grammar Correction.csv', sep=',')

df['label'] = df['Error Type'].map(error_type_to_index)

texts = df['Ungrammatical Statement'].tolist()
labels = df['label'].tolist()
# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenize texts
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_length = 256  # Define maximum length of input
train_tokenized = tokenize_texts(train_texts, tokenizer, max_length)
test_tokenized = tokenize_texts(test_texts, tokenizer, max_length)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(train_tokenized['input_ids'], train_tokenized['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_tokenized['input_ids'], test_tokenized['attention_mask'], torch.tensor(test_labels))

batch_size = 48  # Define batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model
num_classes = len(error_type_to_index)  # Number of classes (positive and negative sentiment)
model = RobertaClassifier(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 30 # Number of training epochs
save_dir = './robertagrammar'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save the model

for epoch in range(epochs):
    model.train()
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            pbar.update(1)  # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        model_path = os.path.join(save_dir, 'roberta_classifier.pt')
        torch.save(model.state_dict(), model_path)
# Evaluation
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

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
