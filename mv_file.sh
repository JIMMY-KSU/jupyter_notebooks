#!/bin/bash

# for f in *[1-9][1-9].png; do echo $f | sed -r 's/([1-9]+)/0\1/g'; done
for f in *p[1-9][0-9].png; do mv "$f" "`echo $f | sed -r 's/([1-9]+)/0\1/g'`"; done
for f in *p[1-9].png; do mv "$f" "`echo $f | sed -r 's/([1-9]+)/00\1/g'`"; done
