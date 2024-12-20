# URLs to scrape and Scraping code

For this project, we initially used the Newspaper3k library to scrape some URLs and extract HTML from documents. We recommend that you utilize the Newspaper4k (https://github.com/AndyTheFactory/newspaper4k) library to scrape and extract article news data from our list of URLs.

We utilize both the htmldate library (https://pypi.org/project/htmldate/) to extract out article data metadata. If htmldate fails, we utilize the date provided by the Newspaper library. 

We additionally provide code for extracting CommonCrawl data for the list of domains that we utilize in this project. To utilize CommonCrawl HTML rather than directly scraping each website, you must first utilize the `common_crawl_indices.py` file in order to get the indices of the WARC files that contain HTML. Secondly, after gathering the WARC index files, you can utilize `html_warc_from_common_crawl_indices.py` to download the WARC files from CommonCrawl. To extract the HTML from these files, you can utilize `warcio` library and  `warcio.archiveiterator import ArchiveIterator` as follows:

```
with open(file ,'rb') as stream:
  for record in ArchiveIterator(stream):
      if record.rec_type == 'response':
          if 'text/html' in record.http_headers.get_header('Content-Type'):
              print(record.rec_headers.get_header('WARC-Target-URI'))
              dom_as_string= str(record.content_stream().read())
```
Similarly to utilizing the Newspaper3k or Newspaper4k library for downloading HTML content, you can subsequently extract article content from URLS using the following snippet of code.
```
article = newspaper.Article(current_url)
article.set_html(dom_as_string)
article.download_state = 2
article.parse()
```

Depending on the website you may have to specify the language of the website as follows.
```
article = newspaper.Article(current_url,language =lang)
```
