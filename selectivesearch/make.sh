#!/bin/bash
make
if [ $? -eq 0 ]; then
    python3 test_chi.py
else
    echo Compliation FAILED
fi
