#!/usr/bin/env python
# encoding: UTF-8

"""Goes through all the inline-links in markdown files and reports the breakages."""

import re
import sys
import pathlib
import os
from multiprocessing.dummy import Pool
from typing import Tuple, NamedTuple, List

import requests


DOC_FILES = [
        "README.md",
        "docs/**/*.md",
]

OK_STATUS_CODES = [200, 403]


http_session = requests.Session()  # pylint: disable=invalid-name
for resource_prefix in ("http://", "https://"):
    http_session.mount(
            resource_prefix,
            requests.adapters.HTTPAdapter(
                    max_retries=5,
                    pool_connections=20,
                    #  pool_maxsize=50,  # doesn't matter since we're not using threads.
            )
    )


class MatchTuple(NamedTuple):
    source: str
    name: str
    link: str


def url_ok(match_tuple: MatchTuple) -> bool:
    """Check if a URL is reachable."""
    try:
        result = http_session.get(match_tuple.link, timeout=5)
        return result.ok or result.status_code in OK_STATUS_CODES
    except (requests.ConnectionError, requests.Timeout):
        return False


def path_ok(match_tuple: MatchTuple) -> bool:
    """Check if a file in this repository exists."""
    relative_path = match_tuple.link.split("#")[0]
    full_path = os.path.join(os.path.dirname(str(match_tuple.source)), relative_path)
    return os.path.exists(full_path)


def link_ok(match_tuple: MatchTuple) -> Tuple[MatchTuple, bool]:
    if match_tuple.link.startswith("http"):
        result_ok = url_ok(match_tuple)
    else:
        result_ok = path_ok(match_tuple)
    print(f"  {'✓' if result_ok else '✗'} {match_tuple.link}")
    return match_tuple, result_ok


def main():
    print("Finding markdown files to check...")

    project_root = (pathlib.Path(__file__).parent / "../..").resolve() # pylint: disable=no-member
    markdown_files: List[pathlib.Path] = []
    for resource in DOC_FILES:
        markdown_files += list(project_root.glob(resource))

    all_matches = set()
    url_regex = re.compile(r'\[([^!][^\]]+)\]\(([^)(]+)\)')
    for markdown_file in markdown_files:
        with open(markdown_file) as handle:
            for line in handle.readlines():
                matches = url_regex.findall(line)
                for name, link in matches:
                    if 'localhost' not in link:
                        all_matches.add(MatchTuple(source=str(markdown_file), name=name, link=link))

    print(f"  {len(all_matches)} markdown files found")
    print("Checking to make sure we can retrieve each link...")

    with Pool(processes=10) as pool:
        results = pool.map(link_ok, [match for match in list(all_matches)])
    unreachable_results = [result for result in results if not result[1]]

    if unreachable_results:
        print(f"Unreachable links ({len(unreachable_results)}):")
        for result in unreachable_results:
            print("  > Source: " + result[0].source)
            print("    Name: " + result[0].name)
            print("    Link: " + result[0].link)
        sys.exit(1)
    print("No unreachable link found.")


if __name__ == "__main__":
    main()
