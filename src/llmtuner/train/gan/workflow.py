# Inspired by: https://github.com/huggingface/transformers/blob/v4.34.1/examples/pytorch/summarization/run_summarization.py
import logging
from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from llmtuner.data import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.model import load_model_and_tokenizer
from llmtuner.train.sft.metric import ComputeMetrics
from llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from llmtuner.train.utils import create_modelcard_and_push
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

class TextDiscriminatorWithTransformer(nn.Module):
    def __init__(self, transformer_model_name, num_classes):
        super(TextDiscriminatorWithTransformer, self).__init__()
        
        # Load pre-trained transformer model and tokenizer
        self.transformer = GPT2Model.from_pretrained(transformer_model_name)
        # Modify architecture as needed (e.g., adding classification layers)
        self.classifier = nn.Sequential(
            nn.Linear(768, num_classes),  # Modify input size based on the transformer's output dimension
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )
        
    def forward(self, x):
        # Tokenize input text
     #   inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        
        # Obtain transformer embeddings
        outputs = self.transformer(**x)
        
        # Use pooled output or hidden states as input to the classifier
        # Here, we're using the pooled output (CLS token)
        
        last_hidden_state = outputs['last_hidden_state']
        
                # Aggregate the hidden states to a single representation for the whole sentence
        aggregated_hidden_state = last_hidden_state.mean(dim=1)  # You can use other aggregation methods as well
        # Apply classification layers
        out = self.classifier(aggregated_hidden_state)      
        return out

# Example usage:
transformer_model_name = "gpt2"  # Change to the specific pre-trained model you want to use
num_classes = 1  # For binary classification

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


def run_gan(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left" # use left-padding in generation

    discriminator = TextDiscriminatorWithTransformer("gpt2", 1)
    generator = GANGenerator()
    
    # Set up optimizers, loss function, and data loader
    optDisc = AdamW(discriminator.parameters(), lr)
    optGen = AdamW(model.parameters(), lr)  # Use the loaded model as the generator
    lossFunc = torch.nn.BCELoss()
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    for epoch in range(numEpochs):

        for idx, (batch,real) in enumerate(dataloader):

            ## training the discriminator here
            fakeData = {} # we construct the fake data, and were going to use it twice
            fakeData["attention_mask"] = batch["attention_mask"].squeeze(1)  #The discriminator will know the right attention mask
            batch["input_ids"] =  batch["input_ids"].squeeze(1)[:,:truncation] # truncating the input
            batch["attention_mask"] = batch["attention_mask"].squeeze(1)[:,:truncation]
            discOutsReal = discriminator(batch)  #tensor like, shaped (batchsize, 1)
            fake = generator.generateText(batch, dataset.maxLength) #tensor like, shaped (batchSize, maxLength)
            fakeData["input_ids"] = fake
            discOutsFake = discriminator(fakeData)
            lossDiscriminatorReal = lossFunc(discOutsReal, torch.ones_like(discOutsReal))   # lossFunc(disc(real), torch.oneslike(disc(real)))
            lossDiscriminatorFake = lossFunc(discOutsFake, torch.zeros_like(discOutsFake))
            finalLoss = (lossDiscriminatorReal + lossDiscriminatorFake) / 2
            if finalLoss > 0.5:
                discriminator.zero_grad()
                finalLoss.backward(retain_graph = True) # adding the retain parameter we ensure that we can use the fake text also for the generator
                optDisc.step()
            
            ## training the generator
            finalLoss = (lossDiscriminatorReal + lossDiscriminatorFake) / 2
            # print("Loss of Discriminator (Real): {:.2f}".format(lossDiscriminatorReal))
            # print("Loss of Discriminator (Fake): {:.2f}".format(lossDiscriminatorFake))
            # print("Final Loss: {:.2f}".format(finalLoss))
            output = discriminator(fakeData) # here the discriminator has been trained once, so this value is different from discOutsFake
            lossGenerator = lossFunc(output, torch.ones_like(output))
            generator.zero_grad()
            lossGenerator.backward()
            optGen.step()
            print("discriminator Loss: {:.2f}".format(finalLoss))
            print("generator Loss: {:.2f}".format(lossGenerator))
            generator.zero_grad()
            
            if idx == 0:
                print("Epoch number ", epoch, " loss Gen: ", lossGenerator, " loss Disc: ", finalLoss)
        
        # once we trained the model for a single epoch, were going to save both models to a local dir

        torch.save(generator.state_dict(), './modelParams/generator' + epoch + ".pth")
        torch.save(discriminator.state_dict(), './modelParams/discriminator' + epoch + ".pth")
