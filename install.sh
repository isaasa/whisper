apt update
apt install ffmpeg -y
apt install python3-pip -y
apt install python3.12-venv -y
python3 -m venv path/to/venv
source path/to/venv/bin/activate
pip3 install -r requirements.txt
cd ..
cp /home/whisper/transru-fp16-del-txt-srt.py /home/
