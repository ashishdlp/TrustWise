
# Step 1: Install the transformers library
# !pip install transformers

# Step 2: Import the necessary libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Step 3: Load the model and tokenizer
# Replace 'your_huggingface_token' with your actual Hugging Face token
model = AutoModelForSequenceClassification.from_pretrained("wajidlinux99/gibberish-text-detector")
tokenizer = AutoTokenizer.from_pretrained("wajidlinux99/gibberish-text-detector")

# Step 4: Tokenize the input text
inputs = tokenizer("Is this text really worth it?", return_tensors="pt")

# Step 5: Get the model's output
outputs = model(**inputs)

# Step 6: Interpret the output
probs = F.softmax(outputs.logits, dim=-1)
print(probs)

# Step 7: Print the probabilities for each class

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Gibberish Detector
gibberish_model = AutoModelForSequenceClassification.from_pretrained("wajidlinux99/gibberish-text-detector")
gibberish_tokenizer = AutoTokenizer.from_pretrained("wajidlinux99/gibberish-text-detector")

# Education Classifier
education_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")
education_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceFW/fineweb-edu-classifier")

def get_all_scores(text):
    # # # Tokenize input text
    gibberish_inputs = gibberish_tokenizer(text, return_tensors="pt")
    education_inputs = education_tokenizer(text, return_tensors="pt")

    # # Get model outputs
    gibberish_outputs = gibberish_model(**gibberish_inputs)
    education_outputs = education_model(**education_inputs)

    # # # Apply softmax to get probabilities
    gibberish_probs = F.softmax(gibberish_outputs.logits, dim=-1)
    education_probs = F.softmax(education_outputs.logits, dim=-1)

    # # Print all scores for each model

    print("\nGibberish Scores:")
    for label, idx in gibberish_model.config.label2id.items():
        print(f"{label}: {gibberish_probs[0][idx].item():.4f}")

    print("\nEducation Scores:")
    for idx, score in enumerate(education_probs[0]):
        print(f"Class {idx}: {score.item():.4f}")


text = "I like you. I love you"
get_all_scores(text)