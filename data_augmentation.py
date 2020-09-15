import nlp
import numpy as np
import pandas as pd
from tqdm import tqdm
from googletrans import Translator
from collections import defaultdict

# Using back translation for data augmentation with just one language

def back_translation(txt_pre, txt_hypo):
    languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',
                 'sw', 'vi', 'es', 'el']
    
    # googletrans translator
    translator = Translator()
    
    # detecting the language of the text
    detected_lang = translator.detect(txt_pre).lang
    b = detected_lang
    
    for _ in range(2):
        # Choosing a random language which is not the detected language and for the second time also for previous random language
        rand_lang = np.random.choice([lang for lang in languages if lang not in [detected_lang, b]])
        
        # Translating Premise and Hypothesis into the random language
        trans_rand_premise = translator.translate(txt_pre, dest = rand_lang).text
        trans_rand_hypothesis = translator.translate(txt_hypo, dest = rand_lang).text
        
        # Converting back to original
        back_trans_premise = translator.translate(trans_rand_premise, dest = detected_lang).text
        back_trans_hypothesis = translator.translate(trans_rand_hypothesis, dest = detected_lang).text
        
        # Checking if the output txt is same as the back translation
        if txt_pre != back_trans_premise:
            return back_trans_premise, back_trans_hypothesis
        
        # Setting b as rand_lang for the second iteration
        b = rand_lang
    
    # returning if the txt is same for both the iteration
    # print ("no change")
    return txt_pre, txt_hypo

def proc(df):
    
    # back_translation dataframe
    df_back = pd.DataFrame(columns = ['premise', 'hypothesis', 'lang_abv', 'language', 'label'])
    
    per = defaultdict(list)
    
    for pre, hypo, lang_abv, language, label in zip(df.premise, df.hypothesis, df.lang_abv, df.language, df.label):
        
        pre_b, hypo_b = back_translation(pre, hypo)
        
        per['premise'].append(pre_b)
        per['hypothesis'].append(hypo_b)
        per['abv'].append(lang_abv)
        per['lang'].append(language)
        per['lab'].append(label)
        
        break
        
    df_back['premise'] = per['premise']
    df_back['hypothesis'] = per['hypothesis']
    df_back['lang_abv'] = per['abv']
    df_back['language'] = per['lang']
    df_back['label'] = per['lab']
    
    df = pd.concat([df_back, df], ignore_index = True)
    
    return df

# Loading the MultiNLI dataset
def load_mnli():
    
    # loading the mnli dataset
    data = nlp.load_dataset(path = 'glue', name = 'mnli')
    
    df_mnli = pd.DataFrame(columns = ['premise', 'hypothesis', 'lang_abv', 'language', 'label'])
    
    df_mnli['premise'] = data['train']['premise']
    df_mnli['hypothesis'] = data['train']['hypothesis']
    df_mnli['label'] = data['train']['label']
    df_mnli['lang_abv'] = ['en']*len(data['train']['premise'])
    df_mnli['language'] = ['English']*len(data['train']['premise'])
    
    return df_mnli

# Loading the XNLI dataset
def load_xnli():
    
    data = nlp.load_dataset(path = 'xnli')
    
    df_xnli = pd.DataFrame(columns = ['premise', 'hypothesis', 'lang_abv', 'language', 'label'])
    
    premise = []
    hypothesis = []
    lang_abv = []
    label = []

    for i in tqdm(range(len(data['test']))):

        premise.extend(list(data['test']['premise'][i].values()))
        hypothesis.extend(list(data['test']['hypothesis'][i].values())[1])
        lang_abv.extend(list(data['test']['hypothesis'][i].values())[0])
        label.extend([data['test']['label'][i]]*15)

    lang_abv_full = {'zh':'Chinese', 'en':'English', 'fr':'French', 'es':'Spanish', 'ar':'Arabic', 'sw':'Swahili', 'ur':'Urdu', 'vi':'Vietnamese',
                     'ru':'Russian', 'hi':'Hindi', 'el':'Greek', 'th':'Thai', 'de':'German', 'tr':'Turkish', 'bg':'Bulgarian'}

    df_xnli['premise'] = premise
    df_xnli['hypothesis'] = hypothesis
    df_xnli['lang_abv'] = lang_abv
    df_xnli['label'] = label
    df_xnli['language'] = df_xnli.lang_abv.replace(lang_abv_full)
    
    return df_xnli