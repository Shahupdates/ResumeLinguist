import os

import requests
import torch
from bs4 import BeautifulSoup
import json
import pandas as pd
import re
import spacy
from gensim import corpora, models
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling, Trainer
from flask import Flask, render_template, request

app = Flask(__name__)

import requests
import pandas as pd
import json


def scrape_usajobs():
    host = 'data.usajobs.gov'
    # Read the credentials from the file
    with open('credentials.txt', 'r') as file:
        credentials = file.read().splitlines()

    if len(credentials) != 2:
        print("Error: credentials file should contain userAgent and authKey on separate lines.")
        return

    userAgent = credentials[0]
    authKey = credentials[1]

    url = 'https://data.usajobs.gov/api/search'
    headers = {
        "Host": host,
        "User-Agent": userAgent,
        "Authorization-Key": authKey,
    }
    params = {
        'JobCategoryCode': '2210',  # this is for Information Technology Management jobs
        'Keyword': 'Software Development',
        'LocationName': 'Washington, DC',
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()

        titles = [item['MatchedObjectDescriptor']['PositionTitle'] for item in
                  data['SearchResult']['SearchResultItems']]
        organizations = [item['MatchedObjectDescriptor']['OrganizationName'] for item in
                         data['SearchResult']['SearchResultItems']]
        locations = [item['MatchedObjectDescriptor']['PositionLocation'][0]['LocationName'] for item in
                     data['SearchResult']['SearchResultItems']]
        descriptions = [item['MatchedObjectDescriptor']['QualificationSummary'] for item in
                        data['SearchResult']['SearchResultItems']]

        job_data = pd.DataFrame({
            'Title': titles,
            'Organization': organizations,
            'Location': locations,
            'Description': descriptions,
        })

        # Perform data cleaning and pre-processing
        job_data["Location"] = job_data["Location"].astype(str).str.replace(r"[\(\[].*?[\)\]]", "").str.strip()

        # Save the data to a CSV file
        job_data.to_csv('usajobs_data.csv', index=False)

        # Check the data
        print("Number of job descriptions:", len(job_data["Description"]))
        print("Number of non-empty job descriptions:", len(job_data[job_data["Description"].str.strip() != ""]))
        print("First few rows of data:")
        print(job_data.head())
        print("Data scraped and saved successfully.")
    else:
        print("Error: Failed to retrieve data from USAJOBS.")


def extract_features():
    # Check if the CSV file with job data exists
    if not os.path.isfile("usajobs_data.csv"):
        print("Error: Job data file 'usajobs_data.csv' not found.")
        return

    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Load the job description data from the CSV file
    data = pd.read_csv("usajobs_data.csv")

    # Check if the job description column exists in the data
    if "Description" not in data.columns:
        print("Error: No job description texts found.")
        return

    # Remove empty job descriptions
    data = data.dropna(subset=["Description"])

    # Check if any job descriptions are available after removing empty rows
    if data.empty:
        print("Error: No valid job description texts found.")
        return

    # Extract key skills using spaCy's Named Entity Recognition (NER)
    skills = []
    for description in data["Description"]:
        doc = nlp(description)
        skills.append([ent.text for ent in doc.ents if ent.label_ == "SKILL"])

    # Extract job titles using spaCy's NER
    titles = []
    for description in data["Description"]:
        doc = nlp(description)
        titles.append([ent.text for ent in doc.ents if ent.label_ == "TITLE"])

    # Perform topic modeling using Gensim
    texts = [description.split() for description in data["Description"]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Check if any terms are available for topic modeling
    if not corpus:
        print("Error: No terms available for topic modeling.")
        return

    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)

    # Get the dominant topic for each job description
    topics = [lda_model[corpus[i]][0][0] for i in range(len(corpus))]

    # Add the extracted features to the DataFrame
    data["Skills"] = skills
    data["Titles"] = titles
    data["Topics"] = topics

    # Save the updated data to a new CSV file
    data.to_csv("usajobs_data_with_features.csv", index=False)

    print("Feature extraction completed and data saved successfully.")


def train_resume_model():
    # Check if the CSV file with features exists
    if not os.path.isfile("usajobs_data_with_features.csv"):
        print("Error: Features data file 'usajobs_data_with_features.csv' not found.")
        return

    # Load the job description data with features from the CSV file
    data = pd.read_csv("usajobs_data_with_features.csv")

    # Check if the necessary columns exist in the data
    if "Titles" not in data.columns or "Description" not in data.columns:
        print("Error: Required columns not found in the features data file.")
        return

    # Combine job titles and job descriptions as model input
    input_data = data["Titles"] + " " + data["Description"]

    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    # Tokenize the input data
    tokenized_data = tokenizer.batch_encode_plus(
        input_data.tolist(),
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir="./resume_generation",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2
    )

    # Create the training dataset
    dataset = TextDataset(
        tokenized_data,
        tokenizer=tokenizer,
        block_size=128
    )

    # Create the data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./resume_generation/fine_tuned_model")

    print("Resume generation model training completed.")



def generate_resume(sample_job_description):
    # Load the fine-tuned GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("./resume_generation/fine_tuned_model")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set the device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Generate a resume for the sample job description
    input_ids = tokenizer.encode(sample_job_description, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, temperature=0.7)

    # Decode the generated resume
    generated_resume = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_resume


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        job_description = request.form["job_description"]
        if job_description:
            generated_resume = generate_resume(job_description)
            return render_template("index.html", generated_resume=generated_resume)

    return render_template("index.html")


if __name__ == "__main__":
    # Call the functions sequentially to scrape data, perform feature extraction, and train the model
    scrape_usajobs()
    extract_features()
    train_resume_model()
    #app.run(debug=True)
