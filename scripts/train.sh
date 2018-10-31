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

if [[ -d /media/data/$USER ]]; then
    default_serialization_dir=/media/data/$USER/models/$model_name
else
    default_serialization_dir=/tmp/models/$model_name
fi

# shellcheck disable=SC2078
while [[ true ]]; do
    #
    # Read base serialization directory from stdin.
    #
    read -e -p "${BOLD}Enter model directory (default is $default_serialization_dir): ${NC}" serialization_dir
    if [[ -z $serialization_dir ]]; then
        serialization_dir=$default_serialization_dir
    fi

    #
    # Check if user has supplied a preexisting run.
    #
    if [[ $serialization_dir =~ ^.*/run_[0-9]+/?$ ]]; then
        if [[ -d $serialization_dir ]]; then
            #
            # In this case, the serialization directory corresponds with a preexisting
            # run, and there are 3 actions the user can take:
            #  1. resume training from the preexisting run,
            #  2. overwrite the preexisting run, or
            #  3. abort and enter a different serialization directory.
            #
            echo -e "${YELLOW}Warning: preexisting run detected.${NC}" >&2
            read -p "${BOLD}Type 'resume' (R), 'overwrite' (O), or 'enter' to abort: ${NC}" action
            if [[ $action =~ ^resume|R$ ]]; then
                echo "Resuming training from preexisting run"
                allennlp_extra_args='--recover'
                break
            elif [[ $action =~ ^overwrite|O$ ]]; then
                #
                # Confirm that user wants to overwrite preexisting run.
                # This will remove everything in that directory.
                #
                echo -e "${YELLOW}Overwriting ${serialization_dir}.${NC}"
                read -p "${BOLD}Is this correct? [Y/n] ${NC}" confirm
                if [[ $confirm =~ ^[Yy]$ ]]; then
                    rm -r "${serialization_dir}"
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
        else
            echo -e "${RED}Error: invalid serialization directory ${serialization_dir}." >&2
            echo -e "Serialization directory should not end with 'run_[0-9]+'.${NC}" >&2
            continue
        fi
    else
        #
        # Get run number.
        #
        if [[ -d $serialization_dir ]]; then
            for file in ${serialization_dir}/run_*; do
                if [[ -d $file ]] && [[ $file =~ ${serialization_dir}/run_[0-9]+ ]]; then
                    (( run_number += 1 ))
                fi
            done
            echo -e "${YELLOW}Detected $run_number previous runs.${NC}"
            (( run_number += 1 ))
        else
            run_number=1
        fi

        #
        # Use subfolder with run number as actual serialization directory.
        #
        run_number=$(printf "%03d" ${run_number})
        serialization_dir=${serialization_dir%%/}/run_${run_number}

        #
        # Confirm.
        #
        echo "Serializing model to ${serialization_dir}"
        read -p "${BOLD}Is this correct? [Y/n] ${NC}" confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            break
        else
            echo -e "${YELLOW}Aborting${NC}" >&2
            continue
        fi
    fi
done

mkdir -p "${serialization_dir}"
SECONDS=0
PYTHONPATH=$ROOTDIR allennlp train "$model_file" \
    --serialization-dir "$serialization_dir" \
    --include-package nlpete.data.dataset_readers \
    --include-package nlpete.models \
    --include-package nlpete.training.metrics \
    $allennlp_extra_args

result=$?
diff=$SECONDS
training_time="$((diff / 3600)) hours, $(((diff / 60) % 60)) minutes and $((diff % 60)) seconds elapsed."
if [[ $result -eq 0 ]]; then
    read -r -d '' message <<- EOM
		✓ Training job \`$model_file\` completed.
		Total time: $training_time
		EOM
else
    read -r -d '' message <<- EOM
		✗ Training job \`$model_file\` failed.
		Total time: $training_time
		EOM
fi
echo "$message"
