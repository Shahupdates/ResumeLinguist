import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import spacy
from gensim import corpora, models
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def scrape_usajobs():
    # Send a GET request to the USAJOBS website
    response = requests.get("https://www.usajobs.gov/Search/Results")

    if response.status_code == 200:
        # Parse the HTML content using Beautiful Soup
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all job listings on the page
        job_listings = soup.find_all("div", class_="job-listing")

        # Create empty lists to store job data
        titles = []
        organizations = []
        locations = []
        descriptions = []

        for listing in job_listings:
            # Extract job title
            title = listing.find("h2", class_="position-title").text.strip()
            titles.append(title)

            # Extract organization name
            organization = listing.find("span", class_="organization-name").text.strip()
            organizations.append(organization)

            # Extract job location
            location = listing.find("span", class_="location").text.strip()
            locations.append(location)

            # Extract job description
            description = listing.find("div", class_="job-description").text.strip()
            descriptions.append(description)

        # Create a DataFrame to store the scraped data
        data = pd.DataFrame({
            "Title": titles,
            "Organization": organizations,
            "Location": locations,
            "Description": descriptions
        })

        # Perform data cleaning and pre-processing
        data["Location"] = data["Location"].str.replace(r"[\(\[].*?[\)\]]", "").str.strip()

        # Save the data to a CSV file
        data.to_csv("usajobs_data.csv", index=False)

        print("Data scraped and saved successfully.")

    else:
        print("Error: Failed to retrieve data from USAJOBS.")


def extract_features():
    # Load the English language model in spaCy
    nlp = spacy.load("en_core_web_sm")

    # Load the job description data from the CSV file
    data = pd.read_csv("usajobs_data.csv")

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
    # Load the job description data with features from the CSV file
    data = pd.read_csv("usajobs_data_with_features.csv")

    # Combine job titles and job descriptions as model input
    input_data = data["Titles"] + " " + data["Description"]

    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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


# Call the functions to scrape data, perform feature extraction, and train the model
scrape_usajobs()
extract_features()
train_resume_model()
