#!/bin/bash

set -e

ROOTDIR="${0%/*}"

all_scripts=( "$@" )
n=${#all_scripts[@]}
i=1
for path in ${all_scripts[@]}; do
    filename=$(basename $path)
    short_name=${path/$ROOTDIR\//}
    echo "Check [$i / $n]: $short_name"
    $path
    (( i += 1 ))
done

exit 0
