#!/bin/bash

for d in */ ; do
    echo "$d"
    find . -type f -exec sh -c 'mv "$0" "$(dirname "$0")/.." ' {} \;
    find . -type d -empty -delete
done