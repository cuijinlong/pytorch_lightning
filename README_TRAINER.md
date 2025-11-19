一、


二、进入FinalShell启动tensorboard
cd /opt/pytorch_lightning/trainer/lightning
conda activate pytorch_lightning
# 重置代理
unset http_proxy && unset https_proxy && unset HTTP_PROXY && unset HTTPS_PROXY
tensorboard --logdir=lightning_logs/ --port 6006 --host 0.0.0.0

三、开放WSL端口映射
netsh interface portproxy show all
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=6006 connectaddress=172.28.237.87 connectport=6006
netsh advfirewall firewall add rule name=WSL2 dir=in action=allow protocol=TCP localport=6006

四、查看显卡状态
# 查看GPU基本信息
nvidia-smi
# 查看更详细的GPU信息
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
# 查看内核消息（包含GPU驱动信息）
dmesg | grep -i nvidia
# 检查NVIDIA驱动版本
nvidia-smi --query-gpu=driver_version --format=csv,noheader
