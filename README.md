#                               llama-chinese 信息抽取之ner实战经验

### 1、用户须知

​		Facebook官方发布的[LLaMA模型禁止商用](https://github.com/facebookresearch/llama)，并且官方没有正式开源模型权重（虽然网上已经有很多第三方的下载地址）。为了遵循相应的许可，目前暂时无法发布完整的模型权重，敬请各位理解（目前国外也是一样）。Facebook完全开放模型权重之后，本项目会及时更新相关策略。**这里发布的是LoRA权重**，可以理解为原LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。本次实验没有进行垂直领域预训练和精调，仅作信息抽取学术研究，尝试大模型在信息抽取方面的可能性，信息抽取在文本生成方面向来比较差，如果信息抽取效果比较好，说明对应的垂直领域对话和文本生成也不会很差。

​      提醒：

   （1）以下中文LLaMA/Alpaca LoRA模型无法单独使用，需要搭配原版LLaMA模型<sup>[1]</sup>。请参考本项目给出的[合并模型](#合并模型)步骤重构模型。

   （2）以下数据和模型等均来自互联网，如果需要商用请找对应的机构商业授权，本次实战经验旨在推动大模型学术研究，尊重原创，侵删。

### 2、中文LLaMA模型

​	中文LLaMA模型在原版的基础上扩充了中文词表，使用了中文纯文本数据进行二次预训练，具体见[训练细节](#训练细节)一节。

| 模型名称          | 类型 |        重构所需模型         | 大小<sup>[2]</sup> |                         LoRA下载地址                         | SHA256<sup>[3]</sup> |
| :---------------- | :--: | :-------------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-LLaMA-7B  | 通用 | 原版LLaMA-7B<sup>[1]</sup>  |        770M        | [[百度网盘]](https://pan.baidu.com/s/1oORTdpr2TvlkxjpyWtb5Sw?pwd=33hb)</br>[[Google Drive]](https://drive.google.com/file/d/1iQp9T-BHjBjIrFWXq_kIm_cyNmpvv5WN/view?usp=sharing) |  39b86b......fe0e60  |
| Chinese-LLaMA-13B | 通用 | 原版LLaMA-13B<sup>[1]</sup> |         1G         | [[百度网盘]](https://pan.baidu.com/s/1BxFhYhDMipW7LwI58cGmQQ?pwd=ef3t)<br/>[[Google Drive]](https://drive.google.com/file/d/12q9EH4mfKRnoKlbkkhzv1xDwWnroo9VS/view?usp=sharing) |  3d6dee......e5199b  |


### 3、中文Alpaca模型

​	中文Alpaca模型在上述中文LLaMA模型的基础上进一步使用了指令数据进行精调，具体见[训练细节](#训练细节)一节。如希望体验类ChatGPT对话交互，请使用Alpaca模型，而不是LLaMA模型。

| 模型名称           |   类型   |        重构所需模型         | 大小<sup>[2]</sup> |                         LoRA下载地址                         | SHA256<sup>[3]</sup> |
| :----------------- | :------: | :-------------------------: | :----------------: | :----------------------------------------------------------: | :------------------: |
| Chinese-Alpaca-7B  | 指令精调 | 原版LLaMA-7B<sup>[1]</sup>  |        790M        | [[百度网盘]](https://pan.baidu.com/s/1xV1UXjh1EPrPtXg6WyG7XQ?pwd=923e)</br>[[Google Drive]](https://drive.google.com/file/d/1JvFhBpekYiueWiUL3AF1TtaWDb3clY5D/view?usp=sharing) |  9bb5b6......ce2d87  |
| Chinese-Alpaca-13B | 指令精调 | 原版LLaMA-13B<sup>[1]</sup> |        1.1G        | [[百度网盘]](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i)<br/>[[Google Drive]](https://drive.google.com/file/d/1gzMc0xMCpXsXmU1uxFlgQ8VRnWNtDjD8/view?usp=share_link) |  45c92e......682d91  |

### 4、Model Hub

​	可以在🤗Model Hub下载以上所有模型，并且使用[transformers](https://github.com/huggingface/transformers)和[PEFT](https://github.com/huggingface/peft)调用中文LLaMA或Alpaca LoRA模型。以下模型调用名称指的是使用`.from_pretrained()`中指定的模型名称。

| 模型名             |            模型调用名称            |                             链接                             |
| ------------------ | :--------------------------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-7B   |  ziqingyang/chinese-llama-lora-7b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-7b) |
| Chinese-LLaMA-13B  | ziqingyang/chinese-llama-lora-13b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-llama-lora-13b) |
| Chinese-Alpaca-7B  | ziqingyang/chinese-alpaca-lora-7b  | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-7b) |
| Chinese-Alpaca-13B | ziqingyang/chinese-alpaca-lora-13b | [Model Hub Link](https://huggingface.co/ziqingyang/chinese-alpaca-lora-13b) |

### 5、脚注及其他说明

**	[1]** 原版LLaMA模型需要[去LLaMA项目申请使用](https://github.com/facebookresearch/llama)或参考这个[PR](https://github.com/facebookresearch/llama/pull/73/files)。因版权问题本项目无法提供下载链接。

**	[2]** 经过重构后的模型大小比同等量级的原版LLaMA大一些（主要因为扩充了词表）。

**	[3]** 下载后务必检查压缩包的SHA256是否一致，完整值请查看[SHA256.md](./SHA256.md)。

压缩包内文件目录如下（以Chinese-LLaMA-7B为例）：

```
chinese_llama_lora_7b/
  - adapter_config.json		# LoRA权重配置文件
  - adapter_model.bin		# LoRA权重文件
  - special_tokens_map.json	# special_tokens_map文件
  - tokenizer_config.json	# tokenizer配置文件
  - tokenizer.model		# tokenizer文件 
```

以下是各原模型和4-bit量化后的大小，转换相应模型时确保本机有足够的内存和磁盘空间（最低要求）：

|                     |   7B   |  13B   |   33B   |   65B   |
| :------------------ | :----: | :----: | :-----: | :-----: |
| 原模型大小（FP16）  | 13 GB  | 24 GB  |  60 GB  | 120 GB  |
| 量化后大小（4-bit） | 3.9 GB | 7.8 GB | 19.5 GB | 38.5 GB |

## 6、合并模型

​	为了将LoRA模型与原版LLaMA进行合并以便进行推理或继续训练，目前提供了两种方式：

- 在线转换：适合Google Colab用户，可利用notebook进行在线转换并量化模型

  [在线转换大模型，这里需要开会员才能下载，推荐手动转换]: https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E5%9C%A8%E7%BA%BF%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2

- 手动转换：适合离线方式转换，生成不同格式的模型，以便进行量化或进一步精调

  [手动转换为大模型]: https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2

  省时妙招和注意事项：

  （1）最好从huggingface上下载*`decapoda-research/llama-7b-hf`*，这个模型是转化后的，不需要用transformers进行转换，可以直接跳过step1.

  在进行模型合并的时候，最好采用huggingface版本的方法merge_llama_with_chinese_lora_to_hf.py，因为这个模型方便后续用transformers方法进行加载。如果huggingface下载比较难受，你们一定感觉难受，可以参照7云盘下载。

  （2）整个训练流程包括词表扩充、预训练和指令精调三部分。词表扩充的代码请参考[merge_tokenizers.py](scripts/merge_tokenizers.py)。

## 7、模型下载

### 7.1、huggingface llama-7b-hf下载，我已经放到云盘上了

​	链接：https://pan.baidu.com/s/1AQM5VVDEERkM75EJB-1gbw?pwd=xzqi  提取码：xzqi

### 7.2、合并后大模型下载，我也放在云盘上了，下载后直接搞微调

​	链接：https://pan.baidu.com/s/16seY6mBhDpT6KPJ120sJPg?pwd=qtnl  提取码：qtnl

## 8、模型训练与验证

### 8.1、数据格式如下：

​	训练集、验证集和测试集处理成处理成这个样子。例如"安装动作_安装&操作部件1_a柱左下部饰板"中''&''表示不同类别和抽取的名词，名词和类别之间用下划线隔开，符号支持自定义，不必照抄。instruction中名词类别部分可以换成你自己的。如果是对话和文本生成，效果应该好于信息抽取，我的数据集相对比较复杂，没有训练的人都不一定标对，哈哈哈。

```
[
    {
        "instruction": "你现在是一个命名实体识别模型，请你帮我抽取出命名实体识别类别为'安装动作','操作部件位置1','方位','目标部件位置1','操作程序选项','一般动作','操作部件位置2','目标部件2','物理量','目标部件位置2','量词','工作区域','拆卸动作','操作程序','目标部件1','操作部件2','操作部件1','一般工具'的二元组，二元组内部用'_'连接，二元组之间用'&'分割。",
        "input": "安装a柱左下部饰板注1个夹子，2个导片；对齐2个前部锁片，然后将卡槽对齐线束支架，接着推动后部夹子，以固定到b柱下部；沿着饰板完全放置好密封条",
        "output": "安装动作_安装&操作部件1_a柱左下部饰板"
    },
    {
        "instruction": "你现在是一个命名实体识别模型，请你帮我抽取出命名实体识别类别为'安装动作','操作部件位置1','方位','目标部件位置1','操作程序选项','一般动作','操作部件位置2','目标部件2','物理量','目标部件位置2','量词','工作区域','拆卸动作','操作程序','目标部件1','操作部件2','操作部件1','一般工具'的二元组，二元组内部用'_'连接，二元组之间用'&'分割。",
        "input": "要安装新的前备箱储物单元，请安装固定到此前备箱储物单元的加固夹。否则，如重复使用前备箱储物单元，请跳至下一步。",
        "output": "安装动作_安装&操作部件1_前备箱储物单元&安装动作_安装&目标部件1_前备箱储物单元&操作部件1_加固夹"
    },
    {
        "instruction": "你现在是一个命名实体识别模型，请你帮我抽取出命名实体识别类别为'安装动作','操作部件位置1','方位','目标部件位置1','操作程序选项','一般动作','操作部件位置2','目标部件2','物理量','目标部件位置2','量词','工作区域','拆卸动作','操作程序','目标部件1','操作部件2','操作部件1','一般工具'的二元组，二元组内部用'_'连接，二元组之间用'&'分割。",
        "input": "拆下将腰部支撑底部固定到座椅靠背框架的螺栓。5nm注此螺栓位于腰部电子控制单元的下部。拆下此螺栓有助于方便地完成以下步骤。提示：推荐使用以下工具：10毫米套筒2英寸加长件棘轮/扭矩扳手",
        "output": "拆卸动作_拆下&操作部件1_螺栓"
    }
]
```

### 8.2、训练命令

```
nohup python finetune.py \
  --base_model './alpaca_combined_hf'\
  --data_path './data/enoch_fine_tune_train.json'\
  --output_dir './output'\ # 存放lora后的模型
```

### 8.3、微调命令

```
python inference_hf.py \
  --lora_model ’./output‘ \ # lora 微调后的模型
  --base_model ‘./alpaca_combined_hf’ \ # 合成后的大模型
  --data_file ’./data/enoch_fine_tune_dev.json‘ \
  --tokenizer_path ’./scripts/merged_tokenizer_hf‘ \# 词表扩充后的目录
```

### 8.3、评价函数

​	metrics.py评价函数，这个函数没有写入到训练验证 过程中，后续有时间在研究如果嵌入到训练和验证部分中去。目前直接用test.py直接加载预测后的结果进行计算得知。

```
import jieba
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import numpy as np
rouge = Rouge()
def compute_metrics(decoded_preds, decoded_labels):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            if pred:
                hypothesis = list(jieba.cut(str(pred)))
                if len(hypothesis) == 0:
                    hypothesis = ['*****']
            else:
                hypothesis = ['*****']
            if label:
                reference = list(jieba.cut(str(label)))
                if len(reference) == 0:
                    reference = ['*****']
            else:
                reference = ['*****']

            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        except Exception as e:
            print(e)
            print(pred)
            print(label)

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict
```

```
import json
from metrics import compute_metrics
ori_list,gen_list = [],[]
with open("/root/autodl-fs/chatglm_finetune/output_dir_pt_20/global_step-300/ft_pt_answer.json", 'r') as f:
    examples = json.loads(f.read())
    for example in examples:
        ori_list.append(example['ori_answer'])
        gen_list.append(example['gen_answer'])
    score_dict = compute_metrics(ori_list,gen_list)
    print(json.dumps(score_dict,ensure_ascii=False,indent=4))
```

```
#信息抽取ner实验结果
{
    "rouge-1": 82.39628086419752,
    "rouge-2": 77.22564362139917,
    "rouge-l": 78.75492695473251,
    "bleu-4": 72.21810946502058
}
```

## 9、本地推理与快速部署

本项目中的模型主要支持以下四种推理和部署方式：

- llama.cpp：提供了一种模型量化和在本地CPU上部署方式
- 🤗Transformers：提供原生transformers推理接口，支持CPU/GPU上进行模型推理
- text-generation-webui：提供了一种可实现前端UI界面的部署方式
- LlamaChat：提供了一种macOS下的图形交互界面

相关文档已移至本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/模型推理与部署)

## 10、局限性

虽然本项目中的模型相比原版LLaMA和Alpaca在中文理解和生成能力上得到显著提升，但也存在以下局限性：

- 可能会产生不可预测的有害内容以及不符合人类偏好和价值观的内容
- 由于算力和数据问题，相关模型的训练并不充分，中文理解能力有待进一步提升
- 暂时没有在线可互动的demo（注：用户仍然可以自行在本地部署）


## 11、引用

如果您觉得本项目对您的研究有所帮助或使用了本项目的代码或数据，请参考引用本项目的技术报告：https://arxiv.org/abs/2304.08177
```
@article{chinese-llama-alpaca,
      title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca}, 
      author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
      journal={arXiv preprint arXiv:2304.08177},
      url={https://arxiv.org/abs/2304.08177},
      year={2023}
}
```

## 12、致谢

本项目基于以下开源项目三次开发，在此对相关项目和研究开发人员表示感谢。

- Facebook LLaMA: https://github.com/facebookresearch/llama
- Chinese-LLaMA-Alpaca by@Yiming Cui: https://github.com/ymcui/Chinese-LLaMA-Alpaca
- Stanford Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora by @tloen: https://github.com/tloen/alpaca-lora
- llama.cpp by @ggerganov: https://github.com/ggerganov/llama.cpp
- pCLUE and translation data by @brightmart: https://github.com/brightmart/nlp_chinese_corpus
- LlamaChat by @alexrozanski: https://github.com/alexrozanski/LlamaChat

Episode: Logo中的小羊驼是由[midjourney](http://midjourney.com)自动生成，并由Mac自带的预览工具自动抠出来的。

## 13、免责声明

**本项目相关资源仅供学术研究之用，严禁用于商业用途。** 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

​	本项目由个人及协作者业余时间发起并维护，因此无法保证能及时回复解决相应问题。支持和尊重原创，侵删。


## 14、问题反馈
如有问题，请在https://github.com/ymcui/Chinese-LLaMA-Alpaca GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 重复以及与本项目无关的issue会被[stable-bot](https://github.com/marketplace/stale)处理，敬请谅解。
- 礼貌地提出问题，构建和谐的讨论社区。
