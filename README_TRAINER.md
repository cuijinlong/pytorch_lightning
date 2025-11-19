一、


二、进入FinalShell启动tensorboard
cd /opt/pytorch_lightning/trainer/lightning
conda activate pytorch_lightning
unset http_proxy && unset https_proxy && unset HTTP_PROXY && unset HTTPS_PROXY
tensorboard --logdir=lightning_logs/ --port 6006 --host 0.0.0.0

三、开放WSL端口映射
netsh interface portproxy show all
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=6006 connectaddress=172.28.237.87 connectport=6006
netsh advfirewall firewall add rule name=WSL2 dir=in action=allow protocol=TCP localport=6006