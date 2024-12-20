"""
common_crawl_indices.py

Description:
------------
This script processes a list of domains to retrieve data from the Common Crawl indices. It uses a custom HTTP adapter to bind requests to a specific source IP address. 
The fetched data is organized into a structured directory format, categorized by domains and Common Crawl index names. 
Domains that have already been processed are logged to avoid redundant operations.

Dependencies:
-------------
- `requests`: For making HTTP requests.
- `publicsuffix2`: For extracting the base public suffix of a domain.
- `argparse`: For parsing command-line arguments.
- `os`, `time`, `json`, `random`: Standard Python libraries for file operations, timing, and randomness.

Command-Line Arguments:
-----------------------
--ip : str
    The source IP address to bind HTTP requests to.

File/Folder Structure:
----------------------
1. Input Files:
    - `path_to_domains_file_1.txt` : File containing a list of domains.
    - `path_to_domains_file_2.txt` : Additional domain list.

2. Output:
    - **Data Directory**:
        `/base_directory/DomainIndices/<domain>/<index>/prefix-<domain>-<index>`
        - Example:
          `DomainIndicies/example.com/CC-MAIN-2023-50/prefix-example.com-CC-MAIN-2023-50`
          
    - **Log File**:
        `completed_domain_files.txt`
        - Logs domains that have already been processed.

How to Run:
-----------
Example usage:
    python script_name.py --ip 192.168.1.10

Steps:
1. Replace input file paths with your actual domain list file paths.
2. Adjust the base directory where fetched data will be stored.
3. Specify the source IP using the `--ip` argument.

Notes:
------
- The script uses a throttling mechanism (`time.sleep`) to avoid overwhelming the Common Crawl API servers.
- Ensure the Public Suffix List (`publicsuffix2` library) is installed and up-to-date.
- Logged domains allow the script to resume processing from where it stopped.
"""

import os
import time
import json
import random
import argparse
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager
from publicsuffix2 import PublicSuffixList

# Custom adapter to set the source IP for HTTP requests
class SourceAddressAdapter(HTTPAdapter):
    def __init__(self, source_address, **kwargs):
        self.source_address = (source_address, 0)
        super(SourceAddressAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, source_address=self.source_address
        )

# Function to validate URLs and ensure "https://" is added
def uri_validator_https_added(url):
    try:
        result = urlparse('https://' + url)
        return all([result.netloc, result.scheme])
    except:
        return False

# Extract domain from a URL
def get_domain(url):
    try:
        domain = urlparse(url).hostname.strip('\r\n')
        return domain.replace("www.", "")  # Remove "www." if present
    except:
        return ""

# Sort a list of items based on length in descending order
def sort_by_length(lst):
    return sorted(lst, key=len, reverse=True)

# Initialize PublicSuffixList
psl = PublicSuffixList()

# Argument parser to set the source IP address
parser = argparse.ArgumentParser(description="Set source IP to use for requests.")
parser.add_argument('--ip', type=str, help='IP address to use for outgoing HTTP requests.')
args = parser.parse_args()
source_ip = args.ip

# Session setup with source IP
session = requests.Session()
session.mount('http://', SourceAddressAdapter(source_ip))
session.mount('https://', SourceAddressAdapter(source_ip))

# Load domains to process
def load_domains(file_paths):
    domains = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            domains.update(file.read().lower().splitlines())
    return domains

# Deduplicate domains using PublicSuffixList
def deduplicate_domains(domains):
    return {psl.get_public_suffix(domain) for domain in domains}

# Main script configuration
if __name__ == "__main__":
    # Example files containing domains
    domain_files = [
        'path_to_domains_file_1.txt',  # Example: 'path_to_domains_file_1.txt'
    ]

    # Load and deduplicate domains
    domains_to_process = deduplicate_domains(load_domains(domain_files))

    # Example List of CommonCrawl indices
    indices = [
        'CC-MAIN-2024-10', 'CC-MAIN-2023-50', 'CC-MAIN-2023-40', 'CC-MAIN-2023-23',
        'CC-MAIN-2023-14', 'CC-MAIN-2023-06', 'CC-MAIN-2022-49', 'CC-MAIN-2022-40'
    ]

    # Completed domains log file
    completed_domains_file = 'completed_domain_files.txt'  
    # Example: 'completed_log_file.txt'
    open(completed_domains_file, 'a+').close()  # Ensure file exists

    # Shuffle domains for randomness
    random.seed(source_ip)
    domains_to_process = list(domains_to_process)
    random.shuffle(domains_to_process)

    # Process each domain
    for domain in domains_to_process:
        with open(completed_domains_file, 'r') as f:
            completed_domains = set(f.read().splitlines())

        if domain in completed_domains:
            continue  # Skip already processed domains

        print(f"Processing domain: {domain} using IP: {source_ip}")

        # Create base directory for domain data
        domain_dir = f'/mnt/projects/qanon_proj/MoreDomainIndices/{domain}'  
        # Example: 'base_directory/DomainIndices/example_domain'
        os.makedirs(domain_dir, exist_ok=True)

        for index in indices:
            try:
                request_url = f'https://index.commoncrawl.org/{index}-index?url=*.{domain}&output=json'
                response = session.get(request_url)
                data_lines = []

                # Parse each JSON line
                for line in response.text.splitlines():
                    try:
                        data_lines.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON line: {e}")

                # Create index-specific directory
                index_dir = f'{domain_dir}/{index}'  
                # Example: 'base_directory/DomainIndices/example_domain/CC-MAIN-2023-50'
                os.makedirs(index_dir, exist_ok=True)

                # Write fetched data to file
                output_file = f'{index_dir}/prefix-{domain}-{index}'  
                # Example: 'base_directory/DomainIndices/example_domain/CC-MAIN-2023-50/prefix-example_domain-CC-MAIN-2023-50'
                with open(output_file, 'w') as f:
                    for data in data_lines:
                        json.dump(data, f)
                        f.write("\n")

                print(f"Fetched {len(data_lines)} records for {domain} in {index}.")
            except Exception as e:
                print(f"Error processing {domain} in index {index}: {e}")
            time.sleep(20)  # Throttle requests to avoid overloading servers

        # Mark domain as completed
        with open(completed_domains_file, 'a') as f:
            f.write(domain + "\n")
