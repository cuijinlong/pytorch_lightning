一、在GPU服务器的宿主机配置端口映射
netsh interface portproxy show all
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=2222
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=2222 connectaddress=172.28.237.87 connectport=2222
netsh advfirewall firewall add rule name=WSL2 dir=in action=allow protocol=TCP localport=2222

二、 在GPU服务器上创建Conda环境
unset http_proxy && unset https_proxy && unset HTTP_PROXY && unset HTTPS_PROXY
conda create --name pytorch_lightning  python==3.10
conda activate pytorch_lightning
三、在本地Pycharm配置GPU服务器上的Conda环境
1、进入 File -> Settings -> Preferences 在左侧菜单中，添加新解释器 -> On SSH
2、配置解释器路径和同步文件夹
（1）Interpreter: 这里需要填写远程服务器上 Python 解释器的路径。你可以通过 SSH 连接到服务器后使用 which python3 或 which python 命令来查找。通常是 /usr/bin/python3 或类似路径。
（2）Sync folders: 这是非常重要的设置。你需要指定一个远程服务器上的目录，用于存放你的项目文件。PyCharm 会自动将本地项目同步到这个远程目录。例如，你可以设置为 /home/your_username/project_name。
（3）勾选 Automatically upload project files to the server，这样当你修改本地文件时，PyCharm 会自动上传。
3、点击 Create。

四、安装依赖包
pip install torch==2.2 torchvision==0.17.0 torchaudio==2.2.0 torchmetrics==0.7.0 lightning==2.2 numpy==1.24.3 pandas==2.3.3 pillow==12.0.0 openpyxl
pip install h5py==3.15.1
pip install tensorboard tqdm
pip install scikit-learn albumentations
pip install litserve
pip install -U openai-whisper
pip install opencv-python==4.8.1.78
pip uninstall opencv-python-headless
pip install numpy==1.24.3
pip install matplotlib
pip install -U ultralytics
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0
pip install timm==0.4.5
pip install rich==13.7.0
pip install rootutils==1.0.7
pip install hydra-colorlog==1.2.0