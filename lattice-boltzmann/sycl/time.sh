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
echo "| 2048 x 2048 |"
echo "|_____________|"
./lbm input_2048x2048.params obstacles_2048x2048.dat
echo "#############################################"
echo "| 4096 x 4096 |"
echo "|_____________|"
./lbm input_4096x4096.params obstacles_4096x4096.dat
echo "#############################################"
