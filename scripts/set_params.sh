#!/bin/bash
set -e
for config in configs/sn_big/*yml
do
    echo $config
    yq eval -i '.training.visualizer_port = ' $config
    #yq eval -i 'del(.training.pretrained)' $config
done
