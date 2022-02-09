#!/bin/zsh

DATA_PATH="data"
PSF_PATH="$DATA_PATH/interim/psf_rgb.png"


automate_capture () {
    name="$1"

    input="$DATA_PATH/raw/$name.jpeg"
    echo ">>> Processing $input"

    capture="$DATA_PATH/captures/$name$2"
    reconstruct="$DATA_PATH/processed/$name$2.png"

    HOST=128.179.164.189
    TMP_PATH="tmp.jpeg"

    #python DiffuserCam/scripts/remote_display.py --fp "$input" --hostname "$HOST" --pad 75 --vshift 20 --brightness 70

    PAD=75
    VSHIFT=20
    BRIGHTNESS=100
    REMOTE_DISPLAY_PATH="~/DiffuserCam_display/test.jpg"

    # preprare image
    python DiffuserCam/scripts/prep_display_image.py --fp $input --pad $PAD --vshift $VSHIFT --brightness $BRIGHTNESS --output_path $TMP_PATH

    # Copy picture to raspberry
    scp "$TMP_PATH" "pi@$HOST:$REMOTE_DISPLAY_PATH"

    # Wait for the picture to be ready
    sleep 10

    # BINARY SEARCH FOR EXPOSURE
    exp_min="0"
    exp_max="1"

    found="false"
    echo ""
    while [ $found = "false" ]; do

        exp=$(printf "%.2f" "($exp_min + $exp_max) / 2.0")
        echo -ne "\r\033[1A\033[0KTesting exposure=$exp in range [$exp_min,$exp_max]"

        bright_out=$(
            python DiffuserCam/scripts/remote_capture.py --exp $exp --iso 100 --bayer --fn "$capture" --hostname "$HOST" --nbits_out 12 |\
            perl -ne  'print /max.*?(\d+)/'
        )
        echo " -> max brightness: $bright_out"

        if [ $bright_out -gt 4000 ]; then
            exp_max=$exp
        elif [ $bright_out -lt 3000 ]; then
            exp_min=$exp
        else
            if [ $exp_max = $exp_min ]; then
                echo "Search failed"
                exit 1
            fi
            found="true"
        fi
    done

    if [ $? ]; then
        automate_analyze $capture $reconstruct
        automate_admm $PSF_PATH $reconstruct
    fi
}

automate_analyze() {
    python DiffuserCam/scripts/analyze_image.py --bayer --fp "$1.png" --gamma 2.2 --bg 1.0 --rg 1.0 --save "$2"

}

automate_admm() {
    python DiffuserCam/scripts/admm.py --psf_fp "$1" --data_fp "$2" --n_iter 30 --single_psf --no_plot --save
}

for i in {01..16}; do
    automate_capture "$i"
done
