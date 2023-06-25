# Resume Linguist

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE.TXT)


Resume Linguist is an NLP-driven project aimed at helping users create optimized and effective resumes. The project utilizes Machine Learning, Natural Language Processing (NLP), and web scraping techniques to generate tailored resumes based on job descriptions.

## Table of Contents
- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Debug Tips](#debug-tips)
- [Code Overview](#code-overview)
- [Future Implementations](#future-implementations)
- [Usage](#usage)
- [License](#license)

## Introduction
The Resume Linguist project aims to automate the resume creation process by analyzing job descriptions, extracting key skills and titles, and generating tailored resumes. It leverages web scraping to collect job data, performs feature extraction using NLP techniques, trains a language model to generate resumes, and provides a web interface for users to interact with the model.

## Setup Instructions
1. Clone the repository: `git clone https://github.com/shahupdates/resumelinguist` & `cd resume-linguist`
2. Set up a virtual environment (optional but recommended): `python -m venv venv` &  `source venv/bin/activate`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the application locally: `python app.py`
5. Access the web application by opening a web browser and navigating to: `http://localhost:5000`


## Tech Stack
The Resume Linguist project utilizes the following tech stack:

- Python: Core programming language
- Flask: Web framework for creating the user interface
- HTML/CSS/JavaScript: Frontend development
- BeautifulSoup: Web scraping library for data collection
- Pandas: Data manipulation and analysis
- Spacy: Natural Language Processing library for feature extraction
- Gensim: Topic modeling and document similarity analysis
- Transformers: NLP library for training and utilizing transformer models
- PyTorch: Deep learning framework for model training and generation

## Troubleshooting
If you encounter any issues or errors during the setup or usage of the project, consider the following troubleshooting tips:

- Ensure that you have installed all the required dependencies specified in the `requirements.txt` file.
- Check that the necessary datasets and files are in the correct locations and have the required permissions.
- Verify that the Flask server is running and accessible at the specified URL.

## Debug Tips
To assist with debugging, consider the following tips:

- Enable debug mode in Flask by setting `app.debug = True` in the `app.py` file.
- Use logging statements to print useful debugging information to the console.
- Check the Flask server logs and any error messages displayed in the web browser's console.

## Code Overview
The project's codebase is structured as follows:

- `app.py`: The main Flask application file that defines the web routes and interactions.
- `scraper.py`: The script for web scraping job data from the USAJOBS website.
- `utils.py`: Utility functions for data cleaning, feature extraction, and model training.
- `templates/`: Directory containing HTML templates for the web interface.
- `static/`: Directory for static files such as CSS stylesheets and JavaScript files.

## Future Implementations
Some potential future enhancements for the Resume Linguist project include:

- Improving the feature extraction process by incorporating additional NLP techniques.
- Implementing user authentication and user-specific resume generation.
- Expanding the model's capabilities to support multiple languages.
- Integrating with professional networking platforms to enhance the job data collection process.

## Usage
1. Access the web application by opening a web browser and navigating to `http://localhost:5000`.
2. Enter a job description and submit the form.
3. The application will generate a tailored resume based on the provided job description.
4. View and copy the generated resume for further use.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
