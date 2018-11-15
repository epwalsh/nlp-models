#!/bin/bash

exit_code=0

trailing_whitespace=$(find ./experiments \
    -name '*.json' \
    -exec egrep -l " +$" {} \;)
for path in $trailing_whitespace; do
    exit_code=1
    echo "  ✗ trailing whitepace: $path"
done

shopt -s globstar
# shellcheck disable=SC2016
empty_line_at_end=$(sed -ns '${/^$/F}' ./experiments/**/*.json)
for path in $empty_line_at_end; do
    exit_code=1
    echo "  ✗ blank line at end of file: $path"
done

exit $exit_code
