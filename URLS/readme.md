# List of URLs to scrape

For this project, we initially used the Newspaper3k library. We recommend that you utilize the Newspaper4k (https://github.com/AndyTheFactory/newspaper4k) library to scrape and extract article news data from our list of URLs.

We utilize both the htmldate library (https://pypi.org/project/htmldate/) to extract out article data metadata. If htmldate fails, we utilize the date provided by the Newspaper library. 
