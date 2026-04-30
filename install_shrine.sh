#!/bin/bash


# git clone shrine 
git clone --depth 1 https://github.com/marcinglowacki/SHRINE.git src/SHRINE/

cd src/SHRINE/

git filter-branch --prune-empty --subdirectory-filter ./python HEAD

touch __init__.py

echo "SHIRNE BASE PYTHON CODE SUCCESFULLY CLONED"

cd ../../

