一、准备数据集
mkdir -p /opt/datasets/pathmnist

sudo chmod +x /Users/cuijinlong/Documents/dev_tools/ffmpeg
mv /Users/cuijinlong/Documents/dev_tools/ffmpeg /usr/local/bin/
ls -la /usr/local/bin/ffmpeg
sudo chmod +x /usr/local/bin/ffmpeg
echo $PATH
which ffmpeg

启动tensorboard
tensorboard --logdir=trainer/runs/resnet_experiment
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=6006 connectaddress=172.28.237.87 connectport=6006
netsh advfirewall firewall add rule name=WSL2 dir=in action=allow protocol=TCP localport=6006