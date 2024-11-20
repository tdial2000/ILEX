#!/bin/bash
msg=${1}


# add new docs
git add build/*/*/*
git add build/*/*
git add source/*/*


# commit
git commit build/*/*/* build/*/* source/*/* -m "$msg"