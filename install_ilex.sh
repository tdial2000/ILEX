#!/bin/bash

ifile="${PWD}/ilex/__init__.py"

# remove __init__.py file in ilex/
if [ -f $ifile ]; then
    rm $ifile
fi 

# create new __init__.py file and add os.env variable to file
touch $ifile
echo "import os" >> $ifile
echo "os.environ['ILEX_PATH'] = '$PWD'" >> $ifile

# run setup.py script
pip install .