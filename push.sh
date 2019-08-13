#!/bin/bash
source build.sh
git add --all
if [ -z "$1"]
then
	git commit -m "push to public"
else
	git commit -m "$1"
fi
git push
