import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

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

        # Create a DataFrame to store the scraped data
        data = pd.DataFrame({
            "Title": titles,
            "Organization": organizations,
            "Location": locations
        })

        # Perform data cleaning and pre-processing
        data["Location"] = data["Location"].str.replace(r"[\(\[].*?[\)\]]", "").str.strip()

        # Save the data to a CSV file
        data.to_csv("usajobs_data.csv", index=False)

        print("Data scraped and saved successfully.")

    else:
        print("Error: Failed to retrieve data from USAJOBS.")


# Call the function to initiate the scraping process
scrape_usajobs()
