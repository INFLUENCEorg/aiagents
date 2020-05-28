#!/bin/bash
SUM=$(cat $1/*.out | grep -o -P "mean=.{0,3}" | cut -c 6- | paste -sd+ | bc)
NLINES=$(cat $1/*.out | grep -o -P "mean=.{0,3}" | wc -l)
AVERAGE=$(bc -l <<< "$SUM / $NLINES")
echo $AVERAGE
