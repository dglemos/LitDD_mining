#!/usr/bin/env python3

import argparse
from datetime import date
from email.utils import parsedate_to_datetime
import os
import subprocess
from lxml import html
from pathlib import Path
import requests


def parse_iso_date(value):
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from exc

def get_file_links(base_url):
    response = requests.get(base_url, timeout=30)
    if response.status_code != 200:
        print(f"Failed to fetch data from {base_url}")
        return []

    page = response.text
    tree = html.fromstring(page)
    file_links = [base_url + link for link in tree.xpath('//a[contains(@href, ".xml.gz") and not(contains(@href, ".md5"))]/@href')]
    return file_links

def get_remote_file_date(file_url):
    response = requests.head(file_url, allow_redirects=True, timeout=30)
    response.raise_for_status()

    last_modified = response.headers.get("Last-Modified")
    if not last_modified:
        return None

    return parsedate_to_datetime(last_modified).date()


def download_files(download_dir, file_links, since_date=None):
    """ Download each file from a list of file URLs """
    for file_url in file_links:
        file_name = file_url.split('/')[-1]
        local_path = os.path.join(download_dir, file_name)

        if since_date is not None:
            remote_date = get_remote_file_date(file_url)
            if remote_date is None:
                print(f"Skipping {file_url}: remote file date is unavailable")
                continue
            if remote_date < since_date:
                print(f"Skipping {file_url}: remote file date {remote_date} is older than {since_date}")
                continue

        # check if file already downloaded 
        if not os.path.exists(local_path):
            print(f"Downloading: {file_url}")
            result = subprocess.call(['wget', '-P', download_dir, file_url])
            if result != 0:
                print(f"Download failed with exit code {result}: {file_url}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=Path, required=True, help="Path to pubmed download directory")
    parser.add_argument("--update_mode", action="store_true", help="Only download PubMed update files from the updatefiles feed")
    parser.add_argument("--since_date", type=parse_iso_date, default=None, help="Only download files whose remote Last-Modified date is on or after YYYY-MM-DD")
    args = parser.parse_args()

    download_dir = os.path.join(args.home_dir, 'raw_download_files')
    os.makedirs(download_dir, exist_ok=True)

    # URLs for PubMed baseline and update files
    PUBMED_BASELINE = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
    PUBMED_UPDATE = 'https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/'

    baseline_files = get_file_links(PUBMED_BASELINE)
    update_files = get_file_links(PUBMED_UPDATE)

    if args.update_mode:
        all_files = update_files
    else:
        all_files = baseline_files + update_files

    download_files(download_dir, all_files, since_date=args.since_date)

if __name__ == "__main__":
    main()
