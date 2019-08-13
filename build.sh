#!/bin/bash
book=${PWD##*/}
cd scripts && python convert.py
cd .. && cd .. && jupyter-book build $book --overwrite && cd $book
