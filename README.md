# L_instruction_learning

## 使用大模型进行指令学习

### 尝试用DeBERTa模型，在训练集上微调后，在test来进行预测，看看具体准确率是多少？

本次任务使用sst2数据集，查看数据格式：

```python
# 加载 SST-2 数据集
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
print(dataset["train"][:1])
print(dataset["test"][:1])
print(dataset["validation"][:1])
```

```
#output:
{'sentence': ['hide new secretions from the parental units '], 'label': [0], 'idx': [0]}
{'sentence': ['uneasy mishmash of styles and genres .'], 'label': [-1], 'idx': [0]}
{'sentence': ["it 's a charming and often affecting journey . "], 'label': [1], 'idx': [0]}
```

```python
print(dataset["test"]["label"][:10]) #test中的标签全部是-1 不是真实标签
```

```
#output:
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
```

所以：test中的标签全部是-1 不是真实标签

#### 任务：

**1.用全部的train训练集微调deberta-base，记录accuracy。**

**2.仅使用16、64、256和1024个样本，微调deberta-base，记录accuracy。**

因为test集的标签全为-1，不是真是标签。所以本次任务用validation集来完成预测并输出为.csv文件。

本次任务环境使用kaggle平台，GPU:P100

完整代码在/sst2_deberta_finetune.py

#### 准确率结果统计：

每次训练选择3个epoch，batch_size为32。以每次训练的最后epoch的准确率进行统计。

| 训练使用的样本数 | 16 | 64 | 256 | 1024 | 全部 |
|-|-|-|-|-|-|
|准确率(accuracy)| 0.4908 | 0.5516 | 0.5998 | 0.9071 | 0.9553 |

&nbsp;

### 指令学习

**要点：**
1. 大模型的智能看起来是通过生成模型的“涌现”机制来完成的，也就是大模型在使用之前经过来类似的过程来进行训练，这样它可以根据上文来预测下文的生成。
   
2. 越大越新的模型可以带来越好的指令学习能力，反之，越小越旧的模型可能会出现“幻觉”(hallucination)。因此，在进行指令学习而不需要进行模型的训练时，尽可能使用更大的模型来完成，例如ChatGPT、GPT-o1、全量的DeepSeek-r1、Qwen等模型。
   
3. 指令模版一般参考alpaca，其书写规范可以参考：https://github.com/tatsu-lab/stanford_alpaca


**询问细节：**

```python
prompt = """Below is an instruction that describes a task,
paired with an input that provides further context. Write a
response that appropriately completes the request.
Before answering, think carefully about the question and
create a step-by-step chain of thoughts to ensure a logical
and accurate response.
### Instruction:
You are an expert of consumer comment analysis. Analyze the given text from
an online review and determine
the sentiment polarity. Return a single number of either -1
and 1, with -1 being negative and 1 being the positive
sentiment.
### Input:
this film 's relationship to actual tension is the same as
what christmas-tree flocking in a spray can is to actual
snow : a poor -- if durable -- imitation .
### Response:
<think>"""
```

1. 在Instruction的第一句，要明确的说明所分析的领域，类似于角色扮演，告诉模型你是什么角色。
   
2. 具体执行的过程放在Instruction部分，给出明确的输出。
 
3. 注意每个section之前的###

4. 注意在Respone之后的<think>标签，这是为了触发模型的Reasoning过程

**根据教程，下面要尝试不同大小的模型，并且完成所有预测统计准确率。试着用Llama、Qwen、Gemma、Phi-4和Mistral来统计下SST-2指令学习的准确度。再和全量微调、小样本微调的DeBERTa结果对比一下。**


经过尝试，Llama（Meta）和 Gemma（Google）模型需要申请：

https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
https://huggingface.co/google/gemma-2-2b-it

暂时不能获取使用。下面统计的是已经预测的结果：



