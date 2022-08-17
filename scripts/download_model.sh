#!/bin/bash
echo "0 - Small version"
echo "1 - Big version"
read -p "Enter the version ID you want to download: " v_id

echo "0 - Car (trained on CompCars)"
echo "1 - Car P3D (trained on Pascal3D+ Cars)"
echo "2 - Bird (trained on CUB-200-2011)"
echo "3 - Moto (trained on LSUN Motorbike)"
echo "4 - Horse (trained on LSUN Horse)"
echo "5 - Synthetic objects (trained on each ShapeNet category)"
read -p "Enter the model ID you want to download: " m_id

mkdir -p models
cd models

if [ $v_id == 0 ]
then

    if [ $m_id == 0 ]
    then
        echo "start downloading car model..."
        gdown 16aIw88ZiAUFUOOBFXdHOUNtJ1-w3zpJG -O car.pkl
        echo "done"

    elif [ $m_id == 1 ]
    then
        echo "start downloading car_p3d model..."
        gdown 1p3ow2LjgrkI3Rcdk-51w2qMg7vxV6doX -O car_p3d.pkl
        echo "done"

    elif [ $m_id == 2 ]
    then
        echo "start downloading bird model..."
        gdown 1nWrmMCjeJzK5nHhZ021CCYS-51LTpKHe -O bird.pkl
        echo "done"

    elif [ $m_id == 3 ]
    then
        echo "start downloading moto model..."
        gdown 1wuVjllVUSVWUyfoleSHd2qKiET-x-l1i -O moto.pkl
        echo "done"

    elif [ $m_id == 4 ]
    then
        echo "start downloading horse model..."
        gdown 1DoJ0HQ60veEPTmWB4JJ_NGQa5U_48Yhs -O horse.pkl
        echo "done"

    elif [ $m_id == 5 ]
    then
        echo "start downloading airplane ShapeNet model..."
        gdown 1WkqfL7zoOrPegHoZCFxy8kTI_DTnFj1W -O sn_airplane.pkl
        echo "done, start downloading bench ShapeNet model..."
        gdown 1__EgJZTtz2y3xI963vgY6j3-kTz8tHaC -O sn_bench.pkl
        echo "done, start downloading cabinet ShapeNet model..."
        gdown 1Yql_enYUniDDP8HXhQ-ZD9hWuvpI6wj6 -O sn_cabinet.pkl
        echo "done, start downloading car ShapeNet model..."
        gdown 1nF_xJfdUsepUkN-i88WaJYpR8RHKCCxI -O sn_car.pkl
        echo "done, start downloading chair ShapeNet model..."
        gdown 1sDdERppgW-q3pCoATCcbVrBNsY5cbPB6 -O sn_chair.pkl
        echo "done, start downloading display ShapeNet model..."
        gdown 1q93zt9cJKO4rrNkQ2NkqqIHikd2xm0LG -O sn_display.pkl
        echo "done, start downloading lamp ShapeNet model..."
        gdown 1kDV9ulT9ip1cQKamauX-YgPFuCnF1w3G -O sn_lamp.pkl
        echo "done, start downloading phone ShapeNet model..."
        gdown 1MpUnyb9w6ZE7_EKUkz35JdADRueW9zDO -O sn_phone.pkl
        echo "done, start downloading rifle ShapeNet model..."
        gdown 1L5TXJldoeoBshgHuPd3rsSAmz_lCMnjt -O sn_rifle.pkl
        echo "done, start downloading sofa ShapeNet model..."
        gdown 1u2Mi4hf2_pfmWVLEcsrekaNcrK-6XOew -O sn_sofa.pkl
        echo "done, start downloading speaker ShapeNet model..."
        gdown 1ZoEOmtnB6aYH05fD0tJba038Wbk1ZLf7 -O sn_speaker.pkl
        echo "done, start downloading table ShapeNet model..."
        gdown 1MwGZpFaadA-3fA1WpXKmX-v7btXcuZJ7 -O sn_table.pkl
        echo "done, start downloading vessel ShapeNet model..."
        gdown 1-2Jwek4GmYDciRNu2K6zsMlyW7c3krBl -O sn_vessel.pkl
        echo "done"

    else
        echo "You entered an invalid model ID!"
    fi

elif [ $v_id == 1 ]
then

    if [ $m_id == 0 ]
    then
        echo "start downloading car model..."
        gdown 1i7HF8EhI--EeES8X8GfN2wDzopulV5Z7 -O car_big.pkl
        echo "done"

    elif [ $m_id == 1 ]
    then
        echo "start downloading car_p3d model..."
        gdown 1N7njgw5tde9wWS6Nwc8WFgJJL61Xu_rs -O car_p3d_big.pkl
        echo "done"

    elif [ $m_id == 2 ]
    then
        echo "start downloading bird model..."
        gdown 1BsUFIYFnrwaMFzW25wx26vg8MS8hZloy -O bird_big.pkl
        echo "done"

    elif [ $m_id == 3 ]
    then
        echo "start downloading moto model..."
        gdown 1A9u-Pmc7UbC2n7Q3bfikFgKNFuKiq9jN -O moto_big.pkl
        echo "done"

    elif [ $m_id == 4 ]
    then
        echo "start downloading horse model..."
        gdown 1V6_XcegVNGHCRRwN2M_IvGL0_bcwhQtV -O horse_big.pkl
        echo "done"

    elif [ $m_id == 5 ]
    then
        echo "start downloading airplane ShapeNet model..."
        gdown 1LHWcswUfMwZpuihb8ZC5IJnvJxZcRpEF -O sn_big_airplane.pkl
        echo "done, start downloading bench ShapeNet model..."
        gdown 1fm5uc_i_KR1fHcdzg3lK3pShtfEdbsnn -O sn_big_bench.pkl
        echo "done, start downloading cabinet ShapeNet model..."
        gdown 1MB32hZrmRBSmMoKmewbiLB51eqSu6-bd -O sn_big_cabinet.pkl
        echo "done, start downloading car ShapeNet model..."
        gdown 1aMPFkXAkDKa9CDy9iX5RTzjDO5N8h0t7 -O sn_big_car.pkl
        echo "done, start downloading chair ShapeNet model..."
        gdown 1pzsZ482Q5utMUeYjehkYpRO3Eo7Rkg4h -O sn_big_chair.pkl
        echo "done, start downloading display ShapeNet model..."
        gdown 1_6I9K3rtv81uj-cRvFT9Nyqn1nBrj6zf -O sn_big_display.pkl
        echo "done, start downloading lamp ShapeNet model..."
        gdown 15Gg4CB7oOWwmKCGcZ8BIyZzA38buVLRM -O sn_big_lamp.pkl
        echo "done, start downloading phone ShapeNet model..."
        gdown 1kyWehfJN8lUojqMXLNTdgDVFt_wDFmbI -O sn_big_phone.pkl
        echo "done, start downloading rifle ShapeNet model..."
        gdown 1FIMAK-hnW8UvEY1YRsJZV7W97_t47_BT -O sn_big_rifle.pkl
        echo "done, start downloading sofa ShapeNet model..."
        gdown 19ZHsc9XbNRi6u3uhMx8s4v0jFSgFw-vf -O sn_big_sofa.pkl
        echo "done, start downloading speaker ShapeNet model..."
        gdown 1YJgDm4EYP33DG2pyUHgHnjajMyLI_WPJ -O sn_big_speaker.pkl
        echo "done, start downloading table ShapeNet model..."
        gdown 1UrQCuk_bDiv9rx4ZAhFTiqSJNUKIf-AW -O sn_big_table.pkl
        echo "done, start downloading vessel ShapeNet model..."
        gdown 1r0Ae3LQGKGDeoGGTwCLj22cMjUnA-B-S -O sn_big_vessel.pkl
        echo "done"

    else
        echo "You entered an invalid model ID!"
    fi

else
    echo "You entered an invalid version ID!"
fi

cd ..
