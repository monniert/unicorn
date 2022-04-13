#!/bin/bash
set -e
for config in configs/sn/*yml
do
    echo $config
    yq eval -i 'del(.model.renderer.shading_type)' $config
    yq eval -i 'del(.model.renderer.simple_shader)' $config
    yq eval -i 'del(.model.renderer.clip_inside)' $config
    yq eval -i 'del(.training.pretrained)' $config
    #yq eval -i '.model.name = "unicorn"' $config
done
