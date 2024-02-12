#!/usr/bin/bash

frb='211127'
MJD0=51559.319
MJD1=59545.002650462964084
F0=11.1946499395
F1=1.5666e-11
DM=234.87

# xfile="/fred/oz002/askap/craft/craco/processing/output/${frb}/htr/${frb}_X_t_${DM}.npy"
# yfile="/fred/oz002/askap/craft/craco/processing/output/${frb}/htr/${frb}_Y_t_${DM}.npy"
xfile="/fred/oz313/processing/output/${frb}/htr/${frb}_X_t_${DM}.npy"
yfile="/fred/oz313/processing/output/${frb}/htr/${frb}_Y_t_${DM}.npy"


python3 /fred/oz002/tdial/ilex/ilexscripts/make_new_stokes_dynspec.py -x $xfile -y $yfile --fast --stks "QUV" --cfreq 1271.5 --f0 1103.5 --bw 336 --bline --baseline 50.0 --guard 5.0 --MJD0 $MJD0 --MJD1 $MJD1 --F0 $F0 --F1 $F1 --ofile "/fred/oz002/tdial/FRBdata/FRB_${frb}/htr/${frb}_100kHz" --nFFT 3360 --tN 150
