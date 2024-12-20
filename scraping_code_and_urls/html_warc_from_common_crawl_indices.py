#!/usr/bin/env python3
# coding: utf-8

"""
Script Description:
-------------------
This script processes a list of domains, verifies their indexed data, and retrieves associated WARC files.
It uses an external script (`warc-retrieval.py`) to extract HTML data from Common Crawl index directories.

Progress is tracked by maintaining a log file to avoid redundant processing of already completed domains.

Command-Line Arguments:
-----------------------
--ip : str
    The source IP address to use for HTTP requests.

Output Structure:
-----------------
1. Indexed Data Directory:
    - DomainIndices/<domain> ## Example
2. HTML Output Directory:
    - DomainHTML/<domain> ## Example
3. Log File:
    - finished_domains.txt
"""

import os
import platform
import random
import subprocess
import argparse

# Argument parser to get source IP
parser = argparse.ArgumentParser(description="Set IP to use for WARC retrieval.")
parser.add_argument('--ip', type=str, help='IP address to use.')
args = parser.parse_args()
source_ip = args.ip

def creation_date(path_to_file):
    """
    Get the creation date of a file. If unavailable (on Linux), return the last modified time.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime  # For MacOS
        except AttributeError:
            return stat.st_mtime  # For Linux fallback to last modified time

def load_domains(file_path):
    """
    Load domain names from a file into a set.
    """
    with open(file_path, 'r') as f:
        return {line.strip() for line in f}

def append_to_log(log_file, domain):
    """
    Append a processed domain to the log file.
    """
    with open(log_file, 'a') as f:
        f.write(domain + "\n")

def get_processed_domains(log_file):
    """
    Get already processed domains from the log file.
    """
    if not os.path.exists(log_file):
        open(log_file, 'a').close()
    with open(log_file, 'r') as f:
        return set(f.read().splitlines())

def process_domain(domain, source_ip):
    """
    Process a single domain: retrieve WARC data using an external script and save to output directories.
    """
    base_input_dir = f'DomainIndices/{domain}'
    base_output_dir = f'DomainHTML/{domain}'

    if not os.path.isdir(base_output_dir):
        os.makedirs(base_output_dir)  # Create output directory if it doesn't exist

    # Iterate through folders in the input directory
    for folder in os.listdir(base_input_dir):
        input_folder = os.path.join(base_input_dir, folder)
        if os.path.isdir(input_folder):
            try:
                # Command to call the WARC retrieval script
                bash_command = (
                    f"python3 warc-retrieval.py "
                    f"-p 1 --ip {source_ip} {input_folder} {base_output_dir}"
                )
                process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
            except Exception as e:
                print(f"Error processing {domain}: {e}")

def main():
    # Paths to necessary files
    indexed_domains_path = 'DomainIndices/'
    completed_domains_file = 'finished_domains.txt'
    domains_file_1 = 'domain_files_to_process.txt'


    # Load domains to process
    indexed_domains = set(os.listdir(indexed_domains_path))
    needed_domains = list(load_domains(domains_file_1))

    # Shuffle domains for randomness
    random.shuffle(needed_domains)

    # Retrieve processed domains from the log
    processed_domains = get_processed_domains(completed_domains_file)

    # Start processing domains
    for domain in all_domains_to_process:
        if domain in processed_domains or domain not in indexed_domains:
            continue  # Skip already processed domains or domains not in indexed directories

        print(f"Processing domain: {domain}")
        num_files = len(os.listdir(f"{indexed_domains_path}/{domain}"))
        if num_files == 0:
            continue  # Skip domains with no files

        # Process the domain
        process_domain(domain, source_ip)
        append_to_log(completed_domains_file, domain)

if __name__ == "__main__":
    main()
