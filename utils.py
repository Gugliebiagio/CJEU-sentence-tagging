from pathlib import Path
import urllib
from tqdm import tqdm
import zipfile
import pandas as pd
import urllib.request
import os
import nltk
from nltk.corpus import wordnet
import string
import numpy as np
import re
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(download_path: Path, url: str):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=download_path, reporthook=t.update_to)

def download_dataset(download_path: Path, url: str):
    print("Downloading dataset...")
    download_url(url=url, download_path=download_path)
    print("Download complete!")

def extract_dataset(zip_path: Path, extract_path: Path):
    print("Extracting all files from the ZIP archive... (it may take a while...)")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed!")

#function that collects the candidates that are similar to the input word
def synonyms(wordgiven: list, ndesired: int):
    #collecting the synsets from the WordNet interface
    wn=wordnet.synsets(wordgiven)
    words=[]
    taglist=[]
    #obtaning the POS tag of the word in process
    wordtag=nltk.pos_tag([wordgiven])[0][1][0]
    check=1
    if len(wn)!=0:
        for word in wn:
            #performing a POS compatibility check on the synset level that result in a prioritization tag
            for entry in word.lemmas():
                if wordtag=='N' and str(entry).split('.')[1]=='n':
                    priortag=1
                elif (wordtag in ['J']) and (str(entry).split('.')[1] in ['a','s']):
                    priortag=1
                elif (wordtag in ['R']) and (str(entry).split('.')[1] in ['r']):
                    priortag=1
                else:
                    priortag=0
                if (wordgiven.lower()!=entry.name().lower()) and (entry.name().lower() not in words):
                    words.append(entry.name().lower()) #list that contains the results
                    taglist.append(priortag) #list that contains the prioritization tags
    #reordering of the results in the list based on their prioritization tag
    wordsfinal=[None]*len(words)
    if len(words)!=0:
        indexstart=taglist.count(1)
        count1=0
        for i in range(0,len(words)):
            if taglist[i]==1:
                wordsfinal[count1]=words[i]
                count1+=1
            else:
                wordsfinal[indexstart]=words[i]
                indexstart+=1
        check=1
    else:
        check=0
    return wordsfinal[0:ndesired],check

#function that determines the most similar word among the ones collected from the WordNet interface via synonyms() function using the GloVe word embeddings
def mostsimilar(embeddings,word,candidates,order=0):
    scores=[]
    cand_score=[]
    check=0
    for element in candidates:
        cand=element.lower()
        #if the candidate contains a "-" or "_" character:
        if (len(cand.split('_'))>1) or (len(cand.split('-'))>1):
            count=0
            sum=0
            #it is replaced by a space character
            cand=cand.replace('_',' ').replace('-',' ')
            #For each part that the candidate is composed of:
            for part in cand.split(' '):
                #the word embedding is found if it exists
                try:
                    sum+=embeddings[part]
                    count+=1
                except:
                    continue
            #if no embedding was found none of the parts the procedure returns the first element on the candidate list given as input
            if (type(sum)==int):
                continue
            else:
                #if there is an existing embedding then the average of embedding that belongs to the part are found and serves as a score
                try:
                  embedding_cand=sum/count
                  cos=(np.dot(embedding_cand,embeddings[word])/np.linalg.norm(embedding_cand))/np.linalg.norm(embeddings[word])
                  scores.append(cos)
                  cand_score.append(element)
                  check+=1
                except:
                  continue
        else:
            #the same procedure
            try:
                embedding_cand=embeddings[cand]
                cos=(np.dot(embedding_cand,embeddings[word])/np.linalg.norm(embedding_cand))/np.linalg.norm(embeddings[word])
                scores.append(cos)
                cand_score.append(element)
                check+=1
            except:
                continue
    #if no embedding was found:
    if len(cand_score)<=order:
        return candidates[0]
    #scores and the corresponding candidates are merged together and resultant list is sorted
    s=np.array([np.array(cand_score),np.array(scores)],dtype=object)
    s=np.transpose(s)
    s=s[np.argsort(s[:,1],axis=0),:][::-1]
    if '_' in s[order][0]:
        s[order][0]=s[order][0].replace('_',' ')
    #first element that has the highest score is returned
    return s[order][0]

#function that cleans the lines from undesired characters as tags,punctuation signs etc.
def cleanline(line):
    tagpattern=r'(<)(.+?)(>)'
    numberpattern=r'([0-9]+)'
    cl=re.sub(tagpattern,'',line.lower()).replace('\n','').replace('“','').replace('”','').replace('…','').replace('—',' ').replace('‘','').replace('’','').replace('\\' , ' ').replace('/',' ')
    cl=re.sub(numberpattern,' ',cl)
    for char in cl:
            if (char in string.punctuation):
                cl=cl.replace(char,' ')
    return cl

def compute_prob_AC(results,train_dyn):
    for i,logits in zip(results.keys(),results.values()):
        probs = torch.nn.functional.softmax(torch.Tensor(logits['prob']), dim=-1).numpy()
        true_class=np.argmax(logits['corr']) #to have probabilities from outputs
        true_class_prob =(probs[true_class])  #the probability of having the right class
        prediction = np.argmax(probs)  #argmax to obtain the predicted class
        is_correct = (prediction == true_class).item() #test if the class is the right one
        if i in train_dyn.keys():

            train_dyn[i]['prob'].append(true_class_prob)
            train_dyn[i]['corr'].append(is_correct)
        else:
            train_dyn[i]={}
            train_dyn[i]['prob']=[true_class_prob]
            train_dyn[i]['corr']=[is_correct]



def compute_prob_TC(results,train_dyn,truth_threshold=0.5):
    # results a dict containing the results obtained by the model
    # train_dyn is a dict updated by this function
    # it contains the probabilities and the true_trend
    # truth_threshold is a float to set the thr. after the sigmoid
    for i,logits in zip(results.keys(),results.values()):

        probs = torch.sigmoid(torch.Tensor(logits['prob'])).numpy() # element-wise sigmoid
        true_class = np.where(np.array((logits['corr'])) == 1)[0] # indexes  of true classes

        true_class_prob =[probs[i] for i in true_class]  # the probability of having the right class
        prediction = np.where(probs > truth_threshold)[0] # predicted classes
        is_correct=check(prediction,true_class)
        if i in train_dyn.keys():
            train_dyn[i]['prob'].append(true_class_prob)
            train_dyn[i]['corr'].append(is_correct)
        else:
            train_dyn[i]={}
            train_dyn[i]['prob']=[true_class_prob]
            train_dyn[i]['corr']=[is_correct]


def compute_prob_SC(results,train_dyn,truth_threshold=0.5):
    # results is a dict containing the results obtained by the model
    # train_dyn is a dict updated by this function
    # it contains the probabilities and the true_trend
    # truth_threshold is a float to set the thr. after the sigmoid
    for i,logits in zip(results.keys(),results.values()):
        probs = torch.sigmoid(torch.Tensor(logits['prob'])).numpy() # element-wise sigmoid
        true_class = np.where(np.array((logits['corr'])) == 1)[0] # indexes  of true classes

        true_class_prob =[probs[i] for i in true_class]  # the probability of having the right class
        prediction = np.where(probs > truth_threshold)[0] # predicted classes
        is_correct=check(prediction,true_class)
        if i in train_dyn.keys():
            train_dyn[i]['prob'].append(true_class_prob)
            train_dyn[i]['corr'].append(is_correct)
        else:
            train_dyn[i]={}
            train_dyn[i]['prob']=[true_class_prob]
            train_dyn[i]['corr']=[is_correct]




def compute_metrics(train_dyn):
    confidence={}
    variability={}
    correctness={}
    variability_func = lambda conf: np.std(conf)
    for i in (train_dyn.keys()):
        correctness[i] = sum(train_dyn[i]['corr'])
        confidence[i] = np.mean(train_dyn[i]['prob'])  # by definition
        variability[i] = variability_func(train_dyn[i]['prob'])
    column_names = ['index',
                    'confidence',
                    'variability',
                    'correctness',
                  ]
    df=pd.DataFrame([[i,
                        confidence[i],
                        variability[i],
                        correctness[i],
                        ] for i in train_dyn.keys()], columns=column_names)
    return df.set_index('index')

def trainer_AC(model, device, optimizer, loss_fn, loader, epochs, train_dyn={}):
    losses = []
    results = {}
    for epoch in range(epochs):
        model.train()
        print(f'EPOCH {epoch + 1}/{epochs}...')
        for _,data in enumerate(loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids).logits

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluation done on train dataset for data cartography purposes
        print('Generation of samples for data cartography...')
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(loader):
                index = data['index']
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids).logits
                index = index.cpu().detach().numpy()
                fin_targets = targets.cpu().detach().numpy()
                fin_outputs = outputs.cpu().detach().numpy()

                for i in range(len(index)):
                    results[index[i]] = {'prob':fin_outputs[i], 'corr':fin_targets[i]}
        compute_prob_AC(results,train_dyn)
    return losses

def trainer_TC(model, device, optimizer, loss_fn, loader, epochs, train_dyn={}):
    losses = []
    results = {}
    for epoch in range(epochs):
        model.train()
        print(f'EPOCH {epoch + 1}/{epochs}...')
        for _,data in enumerate(loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids).logits

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluation done on train dataset for data cartography purposes
        print('Generation of samples for data cartography...')
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(loader):
                index = data['index']
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids).logits
                index = index.cpu().detach().numpy()
                fin_targets = targets.cpu().detach().numpy()
                fin_outputs = outputs.cpu().detach().numpy()

                for i in range(len(index)):
                  results[index[i]] = {'prob':fin_outputs[i], 'corr':fin_targets[i]}
        compute_prob_TC(results,train_dyn)
    return losses

def trainer_SC(model, device, optimizer, loss_fn, loader, epochs, train_dyn={}):
    losses = []
    results = {}
    for epoch in range(epochs):
        model.train()
        print(f'EPOCH {epoch + 1}/{epochs}...')
        for _,data in enumerate(loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids).logits

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Evaluation done on train dataset for data cartography purposes
        print('Generation of samples for data cartography...')
        model.eval()
        with torch.no_grad():
            for _, data in enumerate(loader):
                index = data['index']
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)

                outputs = model(ids, mask, token_type_ids).logits
                index = index.cpu().detach().numpy()
                fin_targets = targets.cpu().detach().numpy()
                fin_outputs = outputs.cpu().detach().numpy()

                for i in range(len(index)):
                  results[index[i]] = {'prob':fin_outputs[i], 'corr':fin_targets[i]}
        compute_prob_SC(results,train_dyn)
    return losses

def evaluation_run(model, loader, device):
    results = []
    for _, data in enumerate(loader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids).logits
        fin_outputs = torch.round(torch.nn.Softmax(dim=-1)(outputs)).cpu().detach().numpy()

        results.extend(fin_outputs)
    return results

def plot_data_map(dataframe: pd.DataFrame,
                  plot_dir: os.path,
                  hue_metric: str = 'correct.',
                  title: str = '',
                  model: str = 'RoBERTa',
                  show_hist: bool = False,
                  max_instances_to_plot = 55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, context='paper')


    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    main_metric = 'variability'
    other_metric = 'confidence'

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    #num_hues=4
    hue_order = sorted(set(dataframe[hue].unique().tolist()))
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30,
                           hue_order=hue_order)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')
    plot.set_title(f"{title}-{model} Data Map", fontsize=17)

    if show_hist:

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=['confidence'], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel('confidence')
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=['variability'], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel('variability')
        plott1[0].set_ylabel('density')

        plot2 = sns.countplot(x="correct.", data=dataframe, ax=ax3, color='#86bf91')
        ax3.xaxis.grid(True) # Show the vertical gridlines

        plot2.set_title('')
        plot2.set_xlabel('correctness')
        plot2.set_ylabel('density')

    fig.tight_layout()
    filename = f'{plot_dir}/{title}_{model}.pdf'# if show_hist else f'figures/compact_{title}_{model}.pdf'
    fig.savefig(filename, dpi=300)

def annotate(df):
  df2=df.copy()
  for i in range(len(df)):
    if df2.iloc[i]['confidence']>0.5 and df2.iloc[i]['variability']<0.15:
      df2.loc[i,'ann']='easy'
    elif df2.iloc[i]['confidence']<0.5 and df2.iloc[i]['variability']<0.15:
      df2.loc[i,'ann']='hard'
    else:
      df2.loc[i,'ann']='ambigous'
  return df2

def check(A,B):
  for a,b in zip(A,B):
    if a!=b:
      return False
  return True

def compute_correctness(trend: List[float]) -> float:
  """
  Aggregate #times an example is predicted correctly during all training epochs.
  """
  return sum(trend)

class Sentences_Dataset(Dataset):
    def __init__(self, df, tokenizer,labels, max_len):
        self.df = df
        self.max_len = max_len
        self.text = df.Text
        self.tokenizer = tokenizer
        self.targets = df[labels].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def hard_to_learn(df,labels,threshold=0.25):
  diz={i:0 for i in labels}  #-> return the percentage of hard to learn sample per class
  for i in range(len(df)):
    if df.iloc[i]['c*(1-v)']<threshold:
      for k in labels:
        if df.iloc[i][k]==1:
          diz[k]+=1
  for j in labels:
    diz[j] /= df[j].sum()
  
  plt.bar(diz.keys(),diz.values(),color='blue')

# Add labels and title
  plt.xlabel('Class')
  plt.ylabel('Value')
  plt.title('Percentage of Hard To Learn')
