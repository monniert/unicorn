#!/bin/bash
echo "0 - ShapeNet NMR"
echo "1 - CUB-200"
echo "2 - Pascal3D+ Cars"
echo "3 - CompCars"
echo "4 - LSUN Horse"
echo "5 - LSUN Motorbike"
read -p "Enter the dataset ID you want to download: " ds_id

mkdir -p datasets
cd datasets

if [ $ds_id == 0 ]
then
    echo "start downloading ShapeNet NMR..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
    echo "done, start unzipping..."
    unzip NMR_Dataset.zip
    mv NMR_Dataset shapenet_nmr
    echo "done"

elif [ $ds_id == 1 ]
then
    echo "start downloading CUB-200..."
    wget https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/ -O CUB_200_2011.tgz
    echo "done, start unzipping..."
    tar -xf CUB_200_2011.tgz
    mv CUB_200_2011 cub_200
    echo "done"

elif [ $ds_id == 2 ]
then
    echo "start downloading Pascal3D+..."
    wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip 
    echo "done, start unzipping..."
    unzip PASCAL3D+_release1.1.zip
    mv PASCAL3D+_release1.1 pascal_3d
    echo "done, start downloading UCMR annotations..."
    wget https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/cachedir-others.tar.gz && tar -vzxf cachedir-others.tar.gz
    mv cachedir/p3d/data/car_val.mat cachedir/p3d/data/car_test.mat  # XXX renamed to correspond to the real test split
    mv cachedir/p3d datasets/pascal_3d/ucmr_anno
    rm -r cachedir
    echo "done"

elif [ $ds_id == 3 ]
then
    echo "start downloading CompCars..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/comprehensive_cars.zip
    echo "done, start unzipping..."
    unzip comprehensive_cars.zip
    mv comprehensive_cars comp_cars
    echo "done"

elif [ $ds_id == 4 ]
then
    mkdir -p lsun
    cd lsun
    echo "start downloading LSUN Horse..."
    wget http://dl.yf.io/lsun/objects/horse.zip -O horse.zip
    echo "done, start unzipping..."
    unzip horse.zip
    echo "done, start downloading indices..."
    wget http://imagine.enpc.fr/~monniert/UNICORN/data/horse_indices.txt
    mv horse_indices.txt horse/indices.txt
    echo "done"
    cd ..

elif [ $ds_id == 5 ]
then
    mkdir -p lsun
    cd lsun
    echo "start downloading LSUN Motorbike..."
    wget http://dl.yf.io/lsun/objects/motorbike.zip -O motorbike.zip
    echo "done, start unzipping..."
    unzip motorbike.zip
    echo "done, start downloading indices..."
    wget http://imagine.enpc.fr/~monniert/UNICORN/data/moto_indices.txt
    mv moto_indices.txt motorbike/indices.txt
    echo "done"
    cd ..

else
    echo "You entered an invalid ID!"
fi

cd ..
