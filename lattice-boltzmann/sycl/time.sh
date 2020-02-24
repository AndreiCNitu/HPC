#!/bin/bash

echo "#############################################"
echo "| 128 x 128 |"
echo "|___________|"
./lbm input_128x128.params obstacles_128x128.dat
echo "#############################################"
echo "| 128 x 256 |"
echo "|___________|"
./lbm input_128x256.params obstacles_128x256.dat
echo "#############################################"
echo "| 256 x 256 |"
echo "|___________|"
./lbm input_256x256.params obstacles_256x256.dat
echo "#############################################"
echo "| 1024 x 1024 |"
echo "|_____________|"
./lbm input_1024x1024.params obstacles_1024x1024.dat
echo "#############################################"
