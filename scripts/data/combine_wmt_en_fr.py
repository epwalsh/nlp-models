#!/usr/bin/env python3

import argparse

from tqdm import tqdm


ENGLISH_PATH = "data/wmt/giga-fren.release2.fixed.en"
FRENCH_PATH = "data/wmt/giga-fren.release2.fixed.fr"
COMBINED = "data/wmt/english_to_french_train.tsv"


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser("combine_wmt_en_fr.py")
    parser.add_argument("--en", type=str, default=ENGLISH_PATH,
                        help="""Path to English file.""")
    parser.add_argument("--fr", type=str, default=FRENCH_PATH,
                        help="""Path to French file.""")
    parser.add_argument("-o", "--out", type=str, default=COMBINED,
                        help="""Path to output file.""")
    opts = parser.parse_args()
    return opts


def main():
    opts = get_opts()
    combined_file = open(opts.out, "w")
    en_file = open(opts.en, "r")
    fr_file = open(opts.fr, "r")
    for en_line, fr_line in tqdm(zip(en_file, fr_file)):
        en_line = en_line.strip('\n')
        combined_line = f"{en_line}\t{fr_line}"
        combined_file.write(combined_line)
    combined_file.close()
    en_file.close()
    fr_file.close()


if __name__ == "__main__":
    main()
