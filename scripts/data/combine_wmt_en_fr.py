#!/usr/bin/env python3

from tqdm import tqdm


ENGLISH_PATH = "data/wmt/giga-fren.release2.fixed.en"
FRENCH_PATH = "data/wmt/giga-fren.release2.fixed.fr"
COMBINED = "data/wmt/english_to_french.tsv"


def main():
    combined_file = open(COMBINED, "w")
    en_file = open(ENGLISH_PATH, "r")
    fr_file = open(FRENCH_PATH, "r")
    for en_line, fr_line in tqdm(zip(en_file, fr_file)):
        en_line = en_line.strip('\n')
        combined_line = f"{en_line}\t{fr_line}"
        combined_file.write(combined_line)
    combined_file.close()
    en_file.close()
    fr_file.close()


if __name__ == "__main__":
    main()
