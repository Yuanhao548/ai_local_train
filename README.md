# ai_local_train
运行配置建议：
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

首先我们需要将上述远程开源仓库 Clone 到本地，可以使用以下命令：
# 进入到数据库盘
cd /root/autodl-tmp
# 打开学术资源加速
source /etc/network_turbo
# clone 上述开源仓库
git clone https://github.com/open-compass/opencompass.git
git clone https://github.com/InternLM/lmdeploy.git
git clone https://github.com/InternLM/xtuner.git
git clone https://github.com/InternLM/InternLM-XComposer.git
git clone https://github.com/InternLM/lagent.git
git clone https://github.com/InternLM/InternLM.git
git clone https://github.com/QwenLM/Qwen.git
# 关闭学术资源加速
unset http_proxy && unset https_proxy