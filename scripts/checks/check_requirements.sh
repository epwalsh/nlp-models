#!/bin/bash
# Checks to make sure no additional requirements are also listed in the AllenNLP requirements.

allennlp_requirements=allennlp/requirements.txt

additional_requirements=$(grep -Ev '^$|^#.*' additional_requirements.txt | sed -E 's/([a-zA-Z]+)(\=|<|>|$).*/\1/')

exit_code=0
for req in $additional_requirements; do
    matches=$(grep "$req" "$allennlp_requirements")
    if [[ ! -z $matches ]]; then
        echo "  ✗ additional requirement ${req} already listed in ${allennlp_requirements}"
        exit_code=1
    else
        echo "  ✓ ${req}"
    fi
done

exit $exit_code
