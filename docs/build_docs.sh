#!/usr/bin/bash

rm ./source/_autosummary/ilex.*

sphinx-apidoc -P -o  ./source ../ilex 

make html