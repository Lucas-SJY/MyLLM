# ***Nano-LLM*** - 从零开始训练大模型

1. #### 下载源码

```Bash
git clone https://github.com/Lucas-SJY/nano-LLM.git
```

2. #### 环境配置

建议环境为：python 3.9 + Torch 2.1.2 + DDP单机多卡训练

最好在ubuntu系统中，先安装pytorch，再使用requirements.txt 安装其余环境

```Bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

先验证pytorch GPU是否安装成功，如果不成功将无法正常运行

```python
import torch
print(torch.cuda.is_available())
#返回True则是成功
```

3. #### 自己训练模型

先下载训练集

1. 在项目根目录下新建dataset目录，并将以下的链接中的文件全部放在dataset目录下

   tokenizer训练集 https://pan.baidu.com/s/1yAw1LVTftuhQGAC1Y9RdYQ?pwd=6666

   Pretrain 训练集 https://pan.baidu.com/s/1-Z8Q37lJD4tOKhyBs1D_6Q?pwd=6666

   SFT微调训练集 https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data/resolve/master/sft_data_zh.jsonl

   DPO训练集（可选）https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main/dpo

2. 在正式训练前运行data_process.py 进行预处理，使数据符合dataloader的格式，数据较多，需要一定的时间

3. 在model/LMConfig.py 中调整配置

   | dim        | 512/768                           | 模型纬度，这个参数如果<512，那么效果会非常差 |
   | ---------- | --------------------------------- | -------------------------------------------- |
   | n_layers   | 8/16                              | Transformer 层数                             |
   | use_moe    | True/False                        | 是否使用MoE                                  |
   | vocab_size | 应当与Tokenizer保持一致，默认4600 | Tokenizer词汇表大小                          |

   

4. 开始训练

   在终端cd到项目根目录后运行如下命令

   N的意思是有几个GPU，

   运行--h可以查看所有能改的训练时的参数，例如num_workers，是读取数据时并行数，这个有可能会引发奇怪的问题

   如果一直爆显存，可以尝试减小batch size

   ```Bash
   torchrun --nproc_per_node N 1-pretrain.py
   ```

   在完成预训练后，开始微调

   微调时，注意在代码中寻找训练开始时的初始权重，要选择前一步模型输出的那个权重

   第一次运行时，选择sft_data_single.csv作为训练数据，进行单轮对话微调

   这个微调需要运行两次，第二次把数据修改成sft_data_multi.csv，进行多轮对话微调

   剩余参数含义与上一步相同

   ```Bash
   torchrun --nproc_per_node N 3-full_sft.py
   ```

   

5. 测试

   要注意修改对应文件中load的权重

   ```Bash
   python 0-eval_pretrain.py #测试预训练模型的接龙效果
   python 2-eval.py #测试模型的对话效果
   ```

   

   

   

   

   

   本项目原自：https://github.com/jingyaogong/minimind

   
