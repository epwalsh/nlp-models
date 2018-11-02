import argparse
import random
from typing import List


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--out-train", type=str, default="../../data/greetings/train.tsv")
    parser.add_argument("--out-validation", type=str, default="../../data/greetings/validation.tsv")
    opts = parser.parse_args()
    return opts


def read_file(fname: str) -> List[str]:
    clean_lines = []
    with open(fname, "r") as lines:
        for line in lines:
            line = line.strip()
            clean_lines.append(line)
    return clean_lines


def generate_data(fname: str,
                  count: int,
                  first_names: List[str],
                  last_names: List[str],
                  templates: List[str]) -> None:
    with open(fname, "w") as out_file:
        for _ in range(count):
            first_name = random.choice(first_names)
            if random.random() < 0.50:
                last_name = random.choice(last_names)
                name = f"{first_name} {last_name}"
            else:
                name = first_name
            template = random.choice(templates)
            source = template.replace("NAME", name)
            target = f"Nice to meet you, {name}!"
            out_file.write(f"{source}\t{target}\n")


def main() -> None:
    opts = get_opts()

    first_names = read_file("../../data/names/first_names.txt")
    last_names = read_file("../../data/names/last_names.txt")
    templates = read_file("../../data/greetings/source_templates.txt")
    generate_data(opts.out_train, opts.train_count,
                  first_names,
                  last_names,
                  templates)
    generate_data(opts.out_validation, opts.validation_count,
                  first_names,
                  last_names,
                  templates)


if __name__ == "__main__":
    main()
