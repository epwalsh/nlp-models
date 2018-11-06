#!/bin/bash

# Project root directory.
ROOTDIR=$(git rev-parse --show-toplevel)

# Color output to make more readable.
RED=$(tput setaf 1)
YELLOW=$(tput setaf 3)
BOLD=$(tput bold)
NC=$(tput sgr0) # No Color

if [[ $# -gt 1 ]]; then
    echo -e "${RED}Error: $0 takes at most 1 argument.${NC}" >&2
    exit 1
elif [[ $# -gt 0 ]]; then
    model_file=$1
fi

# shellcheck disable=SC2078
while [[ true ]]; do
    #
    # Read model definition from stdin.
    #
    if [[ -z $model_file ]]; then
        read -e -p "${BOLD}Enter model file: ${NC}" model_file
    fi
    model_name=$(basename "${model_file}")
    model_ext="${model_name##*.}"
    model_name="${model_name%.*}"
    if [[ -f $model_file ]]; then
        if [[ $model_ext =~ ^json|jsonnet$ ]]; then
            break
        else
            echo -e "${RED}Error: invalid model file." >&2
            echo -e "Model file must be a JSON/HOCON file, but got ${model_ext}.${NC}" >&2
        fi
    elif [[ -d $model_file ]]; then
        echo -e "${RED}Error: invalid model file." >&2
        echo -e "${model_file} is a directory.${NC}" >&2
    else
        echo -e "${RED}Error: invalid model file." >&2
        echo -e "${model_file} does not exist.${NC}" >&2
    fi
    model_file=""
done

# shellcheck disable=SC2078
while [[ true ]]; do
    #
    # Read base serialization directory from stdin.
    #
    read -e -p "${BOLD}Enter serialization directory: ${NC}" serialization_dir
    if [[ -z $serialization_dir ]]; then
        echo -e "${RED}Error: invalid serialization directory ${serialization_dir}." >&2
        continue
    fi

    vocab_directory=${serialization_dir%%/}vocabulary

    #
    # Validate.
    #
    if [[ -d $vocab_directory ]]; then
        echo -e "${YELLOW}Warning: vocabulary directly already exists.${NC}" >&2
        read -p "${BOLD}Type 'overwrite' (O), or 'enter' to abort: ${NC}" action
        if [[ $action =~ ^overwrite|O$ ]]; then
            #
            # Confirm that user wants to overwrite preexisting files.
            # This will remove everything in that directory.
            #
            echo -e "${YELLOW}Overwriting ${vocab_directory}.${NC}"
            read -p "${BOLD}Is this correct? [Y/n] ${NC}" confirm
            if [[ $confirm =~ ^[Yy]$ ]]; then
                rm -r "${vocab_directory}"
                break
            else
                echo -e "${YELLOW}Aborting${NC}" >&2
                continue
            fi
        else
            echo -e "${YELLOW}Aborting${NC}" >&2
            continue
        fi
    elif [[ -f $serialization_dir ]]; then
            echo -e "${RED}Error: ${serialization_dir} already exists and is a file.${NC}" >&2
            continue
    elif [[ -f $vocab_directory ]]; then
            echo -e "${RED}Error: ${vocab_directory} already exists and is a file.${NC}" >&2
            continue
    fi

    #
    # Confirm.
    #
    echo "Serializing vocab to ${serialization_dir}"
    read -p "${BOLD}Is this correct? [Y/n] ${NC}" confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        break
    else
        echo -e "${YELLOW}Aborting${NC}" >&2
        continue
    fi
done

mkdir -p "${serialization_dir}"
SECONDS=0
PYTHONPATH=$ROOTDIR allennlp make-vocab "$model_file" \
    --serialization-dir "$serialization_dir" \
    --include-package nlpete.data.dataset_readers \
    --include-package nlpete.models \
    --include-package nlpete.training.metrics

result=$?
diff=$SECONDS
total_time="$((diff / 3600)) hours, $(((diff / 60) % 60)) minutes and $((diff % 60)) seconds elapsed."
if [[ $result -eq 0 ]]; then
    read -r -d '' message <<- EOM
		✓ Vocab creation for \`$model_file\` completed.
		Total time: $total_time
		EOM
else
    read -r -d '' message <<- EOM
		✗ Vocab creation for \`$model_file\` failed.
		Total time: $total_time
		EOM
fi
echo "$message"
