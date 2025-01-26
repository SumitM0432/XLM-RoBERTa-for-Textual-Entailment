# XLM-RoBERTa-for-Textual-Entailment

## XLM-RoBERTa
The XLM-RoBERTa model was proposed in Unsupervised Cross-lingual Representation Learning at Scale by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook’s RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

#### https://huggingface.co/transformers/model_doc/xlmroberta.html
#### https://huggingface.co/transformers/model_doc/roberta.html

## Textual entailment
Textual entailment (TE) in natural language processing is a directional relation between text fragments. The relation holds whenever the truth of one text fragment follows from another text. In the TE framework, the entailing and entailed texts are termed premise (p) and hypothesis (h), respectively. The relation between premise and hypothesis can be entailment, contradictory or it can be neutral (neither entailment nor contradictory).

## Dataset
I have used three separate datasets Multi-Genre NLI Corpus (MNLI), Cross-Lingual NLI Corpus (XNLI), and the kaggle (Contradictory, My Dear Watson) Dataset. I have incorporated the datasets into one, hence the dataset is a multilingual dataset of 15 different languages. For Data Augmentation I have used the back translation for the kaggle dataset and used all the premises and hypotheses of different languages(15) of the XNLI corpus.

#### MNLI - https://cims.nyu.edu/~sbowman/multinli/
#### XNLI - https://cims.nyu.edu/~sbowman/xnli/
#### Kaggle dataset - https://www.kaggle.com/c/contradictory-my-dear-watson/data
#### Kaggle Exploratory Notebook - https://www.kaggle.com/code/sumitm004/exploratory-data-analysis-mnli-and-xnli

## Requirements

Libraries - pandas, numpy, transformers, nlp, pytorch, tqdm, googletrans and sklearn.


