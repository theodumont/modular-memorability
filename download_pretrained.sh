echo
echo "Downloading CSN..."

if [ ! -d "vmz" ]; then
    git clone https://github.com/facebookresearch/VMZ.git
    mv VMZ/pt/vmz ./
    rm -r VMZ
    # stop CSN after average pooling
    echo "Modifying vmz/models/utils.py file..."
    sed -i '3s/.*/from model.csn import VideoResNet/' ./vmz/models/utils.py
fi

echo
echo "Downloading HRNet..."

MODEL_NAME="ade20k-hrnetv2-c1"
MODEL_PATH="model/pretrained_weights/$MODEL_NAME"
ENCODER="$MODEL_NAME/encoder_epoch_30.pth"
DECODER="$MODEL_NAME/decoder_epoch_30.pth"

# Download model weights
if [ ! -e "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
fi
if [ ! -e "$ENCODER" ]; then
    wget -P "$MODEL_PATH" "http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER"
fi
if [ ! -e "$DECODER" ]; then
    wget -P "$MODEL_PATH" "http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER"
fi

echo
echo "Done!"