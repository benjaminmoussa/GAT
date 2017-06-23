#!/bin/bash 

echo "SETTING UP"

SCRIPT='
sudo apt-get update;
sudo apt-get upgrade;
sudo apt-get install nginx;

sudo sed -i "1s/.*/user ubuntu;/" /etc/nginx/nginx.conf;
sudo sed -i '/# server_names_hash_bucket_size 64;/a server_names_hash_bucket_size 128;' /etc/nginx/nginx/conf;

cd;
mkdir Projects;
cd Projects;
mkdir GAT;
sudo ln -sT ~/Projects/GAT /var/www/html/GAT;
'

ssh -i "aws-ec2-gat1.pem" ubuntu@ec2-52-38-189-7.us-west-2.compute.amazonaws.com "${SCRIPT}"

echo "COPYING NGINX FILES"

scp -i "aws-ec2-gat1.pem" virtual.conf ubuntu@ec2-52-38-189-7.us-west-2.compute.amazonaws.com:~

COPY_SCRIPT="
sudo mv ~/virtual.conf /etc/nginx/conf.d/ 
"

ssh -i "aws-ec2-gat1.pem" ubuntu@ec2-52-38-189-7.us-west-2.compute.amazonaws.com "${COPY_SCRIPT}"

echo "SUCCESSFULLY FINISHED SETUP"
