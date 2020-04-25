#!/bin/sh
echo Downloading And-Bees dataset
wget https://download.pytorch.org/tutorial/hymenoptera_data.zip -O data.zip

echo Extracting it
unzip data
mv hymenoptera_data data
