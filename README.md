# Channel LM Prompting (and beyond)

This includes an original implementation of Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer. "[Noisy Channel Language Model Prompting for Few-Shot Text Classification][paper]" 2021.

<p align="center">
  <img src="img/teaser.png" width="50%" height="50%">
</p>


## Content

1. [Installation](#installation)
2. [Download & Preprocess Data](#download-and-preprocess-data)
3. [Demonstration-based methods](#demonstration-based-methods)
    - [Zero-shot](#zero-shot)
4. [Tuning methods](#tuning-methods)
    - [Prompt tuning](#prompt-tuning)
    - [Head tuning](#head-tuning)
    - [Transformation tuning](#transformation-tuning)
    - [Standard finetuning](#standard-finetuning)

## Installation

```
$ conda create -n lm-prompt python=3.8
$ conda activate lm-prompt
$ conda install pytorch=1.7.1 -c pytorch
$ pip install transformers
```

**To check the data:**
You can see the list of eleven datasets used in the paper by `ls data/k-shot`. Each dataset consists of five different splits based on five different splits (test sets are the same).


## Demonstration-based methods

<p align="center">
  <img src="img/demonstration.png" width="70%" height="70%">
</p>

This section is for methods which does not update any of the model parameters. For details about methods, please see Section 4.1 of the [paper][paper].

### Zero-shot

```
python main.py \
    --task {task_name} \
    --split {dev|test} \
    --data_dir data \
    --out_dir out \
    --gpt2 gpt2-large \
    --do_zeroshot \
    --method {direct|channel}
```

This command will run zero-shot inference using GPT2-large using four different templates (verbalizers) as reported in the paper.

* For "channel", please specify `--method channel`.
* For "direct", please specify `--method direct`.

Useful notes:
* You can adjust `--batch_size` if you run into OOM issue (default is `32`).
* To use GPT2 with different sizes, please use `--gpt2 {gpt2|gpt2-medium|gpt2-xl}`.

## Tuning methods

<p align="center">
  <img src="img/tuning.png" width="70%" height="70%">
</p>

This section is for methods that fully finetune the model parameters (standard finetuning), or update a very limited number of parameters (prompt tuning, head tuning and transformation tuning). For details about the methods, please see Section 4.2 of the [paper][paper].

### Prompt tuning

```
python main.py \
    --task {task_name} \
    --split {dev|test} \
    --data_dir data \
    --out_dir out \
    --gpt2 gpt2-large \
    --method {direct|channel} \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr {0.1|0.01|0.001}
```

* Please note that GPU parallization is implemented for training, but is not implemented for inference.
* Note that, by default, we use the checkpoint that is trained for 100 steps.

### Head tuning

Use `--head_tune` instead of `--prompt_tune` to the command line for the Prompt tuning method. Note that head tuning is only for the direct baseline.

### Transformation tuning

Use `--transform_tune` instead of `--prompt_tune` to the command line for the Prompt tuning method. Note that transformation tuning is only for the direct baseline.

### Standard finetuning

To finetune the entire model parameters, as in typical finetuning, please do not specify any of `--prompt_tune`, `--head_tune` or `--transform_tune`.

For any questions about the paper or the code, or to request pretrained checkpoints, please contact the first author ([email](mailto:cs.washington.edu)) or leave issues.

This also includes implementations of many recent papers studying prompt-based learning. Please make sure to cite corresponding papers when you use implementations of the methods in this repo.
* Brown et al. NeurIPS 2021. "[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)": for zero-shot and concat-based demonstration methods.
* Zhao et al. ICML 2021. "[Calibrate before use: Improving few-shot performance of language models](https://arxiv.org/abs/2102.09690)": for direct++ formulations.
* Holzman et al. EMNLP 2021. "[Surface Form Competition: Why the Highest Probability Answer Isn't Always Right](https://arxiv.org/abs/2104.08315)": for direct++ formulations.
* Lester et al. 2021. "[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)": for prompt tuning methods


[paper]: https://arxiv.org/abs/2108.04106
[lm-bff-code]: https://github.com/princeton-nlp/LM-BFF/blob/main/tools/generate_k_shot_data.py
[lm-bff-paper]: https://arxiv.org/abs/2012.15723
[zhang-paper]: https://arxiv.org/abs/1509.01626
[zhang-data]: http://goo.gl/JyCnZq



