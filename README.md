# Evaluation and Reimplementation of Channel LM Prompting (and beyond)

This includes an original implementation of Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer. "[Noisy Channel Language Model Prompting for Few-Shot Text Classification][paper]" 2021.

<p align="center">
  <img src="img/teaser.png" width="50%" height="50%">
</p>



## Installation

```
$ conda create -n lm-prompt python=3.8
$ conda activate lm-prompt
$ conda install pytorch=1.7.1 -c pytorch
$ pip install transformers
```

**To check the data:**
Please refer to the original implementation for all the datasets and downloading them. This is a forked repo. 
The data that we used is in data.zip Please unzip the file and use task "SST-5" for "CoVid-19 Tweets, Sentiment Analysis" or use task "cite" for "Legal Citation" 

These datasets are downloaded from Kaggle: 
Covid-19 NLP : https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
Legal Citation NLP : https://www.kaggle.com/datasets/shivamb/legal-citation-text-classification

**Extra Methods & Functionality**

Tested the paper on other LLMs including BERT & other datasets to confirm if the results hold true.

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



