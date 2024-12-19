# Across the Firewall: Foreign Media's Role in Shaping Chinese Social Media Narratives on the Russo-Ukrainian War

Authors: Hans W. A. Hanley, Yingdan Lu, Jennifer Pan

There is a widespread perception that China's digital censorship distances its people from the global internet, and the Chinese Communist Party, through state-controlled media, is the main gatekeeper of information about foreign affairs. Our analysis of narratives about the Russo-Ukrainian War circulating on the Chinese social media platform Weibo challenges this view. Comparing narratives on Weibo with 8.26 million unique news articles from 2,500 of the most trafficked websites in China, Russia, Ukraine, and the US (totaling 10,000 sites), we find that Russian news websites published more articles matching narratives found on  Weibo than news websites from China, Ukraine, or the US. Similarly, a plurality of Weibo narratives were most associated with narratives found on Russian news websites while less than ten percent were most associated with narratives from Chinese news sites. Narratives later appearing on Weibo were more likely to first appear on Russian rather than Chinese, Ukrainian, or US news websites, and Russian websites were highly influential for narratives appearing on Weibo. Altogether, these results show that Chinese state media was not the main gatekeeper of information about Russia's invasion of Ukraine for Weibo users. 

## Website and URL Information

We provide code to scrape and extract metadata for our list of URLs in the URLS folder. We identified most of the URL data for each website from the CommonCrawl data indexed at CC-MAIN-2023-06, CC-MAIN-2022-49, CC-MAIN-2022-40 CC-MAIN-2022-33, CC-MAIN-2022-27, CC-MAIN-2022-21, CC-MAIN-2022-05. To identify popular and influential Chinese, Russian, and Ukrainian news websites, we combined Amazon Alexa Popularity Ranking data  and Common Crawl's Domain rank datasets~\citep{Nagel2017} with website categorization data from Cloudflare.  Namely, we collected the set of most popular websites ranked in Amazon Alexa's top one million websites and Common Crawl's Domain Rank datasets from April 2022, which utilize the top-level domain of each country we were interested in. We then queried Cloudflare's Domain Intelligence API to collect the Cloudflare label for each domain and subsequently gathered domains labeled by Cloudflare as news-related. We collected the domain-registration data of each of the 10,000 domains to confirm that these websites were registered in the respective countries using whois in Python. 

## Clustering and Identifying Weibo Narratives

We first embedded (calculated vector representations) each of our Weibo passages into a shared embedding space, so that passages conveying similar content or ideas have high cosine similarity. To do this, we utilized a monolingual version of Mandarin-Chinese BERT trained on semantic similarity tasks (https://huggingface.co/shibing624/text2vec-base-chinese). We utilize the clustering algorithm provided in the clustering folder to identify different narratives setting the λ parameter utilized for this algorithm to 0.90.

## Multilingual Paraphrasing

To identify paraphrases between different languages, we initially utilize the MP-Net paraphrase transformer (https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) and subsequently utilize three trained models that utilize  Machine Translation (WMT-19) dataset of news commentary bitexts in Chinese and English, and in Chinese and Russian. We provide the architecture and weight for these models in the paraphrasing_model folder. 



## License and Copyright

Copyright 2024 The Board of Trustees of The Leland Stanford Junior University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

