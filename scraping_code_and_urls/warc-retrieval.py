#!/usr/bin/env python
from os import listdir, makedirs
from os.path import join
import json
import zlib
import gzip
import errno
import logging
import requests
from multiprocessing import Pool
from argparse import ArgumentParser
from hashlib import md5
import sys
import itertools
import hashlib
import os
import argparse
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.connection import allowed_gai_family
from requests.packages.urllib3.poolmanager import PoolManager
import requests as re
import json
import argparse


class SourceAddressAdapter(HTTPAdapter):
    def __init__(self, source_address, **kwargs):
        self.source_address = (source_address, 0)
        self.allowed_gai_family = allowed_gai_family
        super(SourceAddressAdapter, self).__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       source_address=self.source_address)



URL_PREFIX = 'https://data.commoncrawl.org/'
DIR_OUTPUT = None
URL_STRIP = '.com/'
CURRENT_IP = ''

def retrieve_indexed_text(index):
    name_output = index['url'][(index['url'].find(URL_STRIP) + len(URL_STRIP)):].replace('/', '-')[:50] + '.warc'
    try:
        source_ip = CURRENT_IP
        s = requests.Session()
        s.mount('http://', SourceAddressAdapter(source_ip))
        s.mount('https://', SourceAddressAdapter(source_ip))
        byte_start = int(index['offset'])
        byte_end = byte_start + int(index['length']) - 1
        r = s.get(URL_PREFIX + index['filename'],
                         headers={'Range': 'bytes=%d-%d' % (byte_start, byte_end)})

        name_output = index['url'][(index['url'].find(URL_STRIP) + len(URL_STRIP)):].replace('/', '-')[:50] + '.warc'
        
        hash_object = hashlib.sha256(name_output.encode())
        hex_dig = hash_object.hexdigest()
        hex_dir = str(hex_dig)[:2]
        name_output = hex_dir+'/'+name_output
        if not os.path.isdir(join(DIR_OUTPUT, hex_dir)):
            os.mkdir(join(DIR_OUTPUT, hex_dir))
        with open(join(DIR_OUTPUT, name_output), 'wb') as f:
            f.write(zlib.decompress(r.content, 32 + zlib.MAX_WBITS))
            #f.write(r.content)
        logging.info('Finished retrieving indexed text ' + name_output)
    except Exception as e:
        logging.info('Abort %s: error when retrieving file; %s' % (name_output, str(e)))


def do_work(dir_index, num_processes):
    """
    :param dir_index: path of directory containing index files
    :param num_processes: the number of processes to use
    :return:
    """
    dict_indices = {}  # Use dict to remove duplicates caused by http/https
    for idx_file in listdir(dir_index):
        if not idx_file.startswith('.'):
            with open(join(dir_index, idx_file), 'r') as f:
                for line in f:
                    try:
                        index = json.loads(line)
                    except:
                        continue
                    key = index['url'][index['url'].find('://'):]
                    dict_indices[key] = index
                if len(dict_indices) > 1000000:
                    indices = dict_indices.values()
                    logging.info('Start to retrieve %d indexed text in total' % len(indices))
                    with Pool(processes=num_processes) as pool:
                        pool.map(retrieve_indexed_text, indices)
                    logging.info('Finished retrieving all %d indexed text' % len(indices))
                    dict_indices = {}
                    
    indices = dict_indices.values()
    logging.info('Start to retrieve %d indexed text in total' % len(indices))
    with Pool(processes=num_processes) as pool:
        pool.map(retrieve_indexed_text, indices)
    logging.info('Finished retrieving all %d indexed text' % len(indices))


def get_args():
    logging.basicConfig(format='%(asctime)s: [%(levelname)s]: %(message)s', level=logging.INFO)

    parser = ArgumentParser('CDX Index Text Retrieval')
    parser.add_argument('dir_index', help='The path of directory containing index files')
    parser.add_argument('dir_output', help='The path of output directory')
    parser.add_argument('-p', '--processes', type=int, default=2,
                        help='Number of worker processes to use; default is 2')
    parser.add_argument('--ip', type=str, help='IP to use.')

    parser.add_argument('--strip', default='.com/',
                        help='Use stripped url as file name of retrieved text; ' +
                             'default is to strip everything before .com/')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    DIR_INDEX = args.dir_index
    DIR_OUTPUT = args.dir_output
    URL_STRIP = args.strip
    CURRENT_IP = args.ip

    try:
        makedirs(DIR_OUTPUT)
    except OSError as e:  # Avoid race condition when creating directory
        if e.errno != errno.EEXIST:
            raise

    do_work(DIR_INDEX, args.processes)warc-retrieval.py
