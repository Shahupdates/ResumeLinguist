import requests
from bs4 import BeautifulSoup


def scrape_job_descriptions(url):
    # Send a get request to the website
    response = requests.get(url)

    # Ensure that the request was successful
    if response.status_code != 200:
        return 'Request failed with status code {}'.format(response.status_code)

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find job descriptions. This might change depending on the structure of the website.
    job_descriptions = soup.find_all('div', {'class': 'jobDescription'})

    # Create a list to store all the job descriptions
    descriptions = []

    for job in job_descriptions:
        descriptions.append(job.text)

    return descriptions


url = 'https://www.indeed.com/'  # replace with the URL of the website you want to scrape
descriptions = scrape_job_descriptions(url)
