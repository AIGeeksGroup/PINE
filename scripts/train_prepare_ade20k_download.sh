#!/usr/bin/env bash
# Download ADE20K into OUTPUT_DIR: bash THIS OUTPUT_DIR
# ADEChallengeData2016 holds the 20,210 training and 2,000 validation images and their 150-class semantic maps;
# annotations_instance holds the object instance masks of the 100 thing classes (Places Challenge 2017).
set -e
mkdir -p "$1"
cd "$1"
curl -sSL -o ADEChallengeData2016.zip http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
curl -sSL -o annotations_instance.tar http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
unzip -q ADEChallengeData2016.zip
tar -xf annotations_instance.tar
rm ADEChallengeData2016.zip annotations_instance.tar
ls
