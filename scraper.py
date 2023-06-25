import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import spacy
from gensim import corpora, models

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


# Call the functions to scrape data and perform feature extraction
scrape_usajobs()
extract_features()
