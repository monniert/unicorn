#!/bin/bash
echo "0 - Car (trained on CompCars)"
echo "1 - Bird (trained on CUB-200)"
echo "2 - Moto (trained on LSUN Motorbike)"
echo "3 - Horse (trained on LSUN Horse)"
echo "4 - Synthetic objects (trained on each ShapeNet category)"
read -p "Enter the model ID you want to download: " m_id

mkdir -p models
cd models

if [ $m_id == 0 ]
then
    echo "start downloading car model..."
    gdown --id 16aIw88ZiAUFUOOBFXdHOUNtJ1-w3zpJG -O car.pkl
    echo "done"

elif [ $m_id == 1 ]
then
    echo "start downloading bird model..."
    gdown --id 1nWrmMCjeJzK5nHhZ021CCYS-51LTpKHe -O bird.pkl
    echo "done"

elif [ $m_id == 2 ]
then
    echo "start downloading moto model..."
    gdown --id 1wuVjllVUSVWUyfoleSHd2qKiET-x-l1i -O moto.pkl
    echo "done"

elif [ $m_id == 3 ]
then
    echo "start downloading horse model..."
    gdown --id 1DoJ0HQ60veEPTmWB4JJ_NGQa5U_48Yhs -O horse.pkl
    echo "done"

elif [ $m_id == 4 ]
then
    echo "start downloading airplane ShapeNet model..."
    gdown --id 1WkqfL7zoOrPegHoZCFxy8kTI_DTnFj1W -O sn_airplane.pkl
    echo "done, start downloading bench ShapeNet model..."
    gdown --id 1__EgJZTtz2y3xI963vgY6j3-kTz8tHaC -O sn_bench.pkl
    echo "done, start downloading cabinet ShapeNet model..."
    gdown --id 1Yql_enYUniDDP8HXhQ-ZD9hWuvpI6wj6 -O sn_cabinet.pkl
    echo "done, start downloading car ShapeNet model..."
    gdown --id 1nF_xJfdUsepUkN-i88WaJYpR8RHKCCxI -O sn_car.pkl
    echo "done, start downloading chair ShapeNet model..."
    gdown --id 1sDdERppgW-q3pCoATCcbVrBNsY5cbPB6 -O sn_chair.pkl
    echo "done, start downloading display ShapeNet model..."
    gdown --id 1q93zt9cJKO4rrNkQ2NkqqIHikd2xm0LG -O sn_display.pkl
    echo "done, start downloading lamp ShapeNet model..."
    gdown --id 1kDV9ulT9ip1cQKamauX-YgPFuCnF1w3G -O sn_lamp.pkl
    echo "done, start downloading phone ShapeNet model..."
    gdown --id 1MpUnyb9w6ZE7_EKUkz35JdADRueW9zDO -O sn_phone.pkl
    echo "done, start downloading rifle ShapeNet model..."
    gdown --id 1L5TXJldoeoBshgHuPd3rsSAmz_lCMnjt -O sn_rifle.pkl
    echo "done, start downloading sofa ShapeNet model..."
    gdown --id 1u2Mi4hf2_pfmWVLEcsrekaNcrK-6XOew -O sn_sofa.pkl
    echo "done, start downloading speaker ShapeNet model..."
    gdown --id 1ZoEOmtnB6aYH05fD0tJba038Wbk1ZLf7 -O sn_speaker.pkl
    echo "done, start downloading table ShapeNet model..."
    gdown --id 1MwGZpFaadA-3fA1WpXKmX-v7btXcuZJ7 -O sn_table.pkl
    echo "done, start downloading vessel ShapeNet model..."
    gdown --id 1-2Jwek4GmYDciRNu2K6zsMlyW7c3krBl -O sn_vessel.pkl
    echo "done"

else
    echo "You entered an invalid ID!"
fi

cd ..
