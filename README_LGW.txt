
### 使用 pyenv

`pyenv` 是一个流行的版本管理工具，可以让你安装多个 Python 版本并轻松切换。安装 `pyenv` 并使用它来安装 Python 3.8 是一个安全、方便的做法：
# 安装依赖
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# 安装 pyenv
curl https://pyenv.run | bash

# 配置环境变量
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# 使用 pyenv 安装 Python 3.8
pyenv install 3.8.10
pyenv global 3.8.10





pretraining:
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py     --stage pt     --do_train     --model_name_or_path ChatGLM3     --dataset wiki_demo     --finetuning_type lora     --lora_target query_key_value     --output_dir ChatGLM3_pt_checkpoint     --overwrite_cache     --per_device_train_batch_size 4     --gradient_accumulation_steps 4     --lr_scheduler_type cosine     --logging_steps 10     --save_steps 1000     --learning_rate 5e-5     --num_train_epochs 3.0     --plot_loss     --fp16

sft:
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ChatGLM3 \
    --adapter_name_or_path ChatGLM3_pt_checkpoint \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir  ChatGLM3_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16





httpx.InvalidURL: Invalid port:
httpx包的bug…

改源码的这个文件lib/python3.9/site-packages/httpx/_urlparse.py：149行左右：

def urlparse(url: str = "", **kwargs: typing.Optional[str]) -> ParseResult:
    url = url.replace("::", ":").replace("[","").replace("]","")
    # Initial basic checks on allowable URLs.
    # ---------------------------------------

    # Hard limit the maximum allowable URL length.
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURL("URL too long")
1
2
3
4
5
6
7
8
加了一行

url = url.replace("::", ":").replace("[","").replace("]","")
1
就行了
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/qq_37668436/article/details/130385526










实测：ValueError: FP16 Mixed precision training with AMP or APEX (`--fp16`) and FP16 h alf precision

2. 解决方法
降低transformers版本
pip install transformers==4.30.1
设置torch版本
pip install torch==2.0.0

经过上面两步后，解决了这个问题。


