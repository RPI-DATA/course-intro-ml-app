#!/bin/bash
book=${PWD##*/}
cd ../; jupyter-book build $book --overwrite; cd $book
git add --all
if [ -z "$1"]
then 
	git commit -m "push to public"
else
	git commit -m "$1"
fi
git push