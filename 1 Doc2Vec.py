import nltk
import sklearn
import numpy as np
import pandas as pd
import re
from gensim.models import doc2vec
import gensim
import pickle
import random
import concurrent.futures

dfa = pd.read_csv('/home/lanna/Dropbox/Insight/X-Cite/df_PLOS.csv')    #PLOS articles
dfb = pd.read_csv('/home/lanna/Dropbox/Insight/X-Cite/df_BioMed.csv')  #BioMed pub articles

df = pd.concat([dfa, dfb])

#Which do not have DOIs?
#nodoi = pd.DataFrame(data = {'Reference':list(set(df.Reference[pd.isnull(df.DOI)])), 'DOI': ""})
#nodoi.to_csv("/home/lanna/Dropbox/Shared/beadyeyes/findDOI.csv",index=False)
#Make it so that:
#1. Tags are Ref#'s that can later be used to index either a DOI or reference

#Remove rows with NaNs for Reference
df = df.dropna(subset=['Reference'])
####################################################
#   1. Clean-up sentences
####################################################
#Clean References:
#Remove PubMed, View Article and PubMed Central at the end of the citations 
target = []
for i in df['Reference']:
    m = re.search(r"(PubMed|View Article|PubMed Central)", i)
    if not not m:
        target.append(i[:m.start(1)])
    else:
        target.append(i)

target2 = []
for i in target:
    tmp = re.sub(r"\n","",i)    #Get rid of extra \n
    tmp = re.sub(r"(doi\: *10\.+)","",tmp)    #Get rid of everything after "doi:"
    tmp = re.sub(r"(pmid\: \d+)","",tmp) #Get rid of everything after "pmid:"
    tmp = re.sub(r"(pmid\:\d+)","",tmp) #Get rid of everything after "pmid:"
    tmp = re.sub(r"(http\:\/\/.+)","",tmp) #Get rid of everything after "http://"
    tmp = re.sub(r"(10\.\d+\/.+)","",tmp)   #Get rid of evertyhing after a doi number of some sort
    tmp = re.sub(r"(10\.\\u\d+\/.+)","",tmp)   #Get rid of evertyhing after a doi number of some sort
    tmp = ' '.join(tmp.split())  #Get rid of extra spaces
    target2.append(tmp)


df['Reference'] = target2   #Add cleaned Reference to dataframe

#############################################################
#Make it so that:
#1. Tags are Ref#'s that can later be used to index either a DOI or reference
#############################################################
#Create a unique ID for each DOI
tmp = ["ID_"+str(i) for i in list(range(0,len(set(df.DOI))))]
byDOI = pd.DataFrame(data = {'ID': tmp,'DOI': list(set(df.DOI))})

#Merge with main data frame to create df1
df1 = pd.merge(byDOI,df, on='DOI')

#If DOI = NaN; or, if ID = ID_0
num = len(set(df1.ID))
tt = list(set(df1.Reference[pd.isnull(df1.DOI)]))
for i in range(0,len(tt)):
    df1.ID[df1.Reference == tt[i]] = "ID_"+str(num)
    num+=1

#*
df1['freq'] = df1.groupby('Reference')['Reference'].transform('count')  #Add frequency of citations to data frame

#################################################
# VALIDATION SET
#################################################
#Select 1,000 of each category for Test Set
def rand_smpl(mylist,num=1000):
    rand_smpl = [ mylist[i] for i in sorted(random.sample(range(len(mylist)), num)) ]
    return rand_smpl

s1 = rand_smpl(list(set(df1[df1['freq']==2].Reference)),1000)
s2 = rand_smpl(list(set(df1[df1['freq']==3].Reference)),1000)
s3 = rand_smpl(list(set(df1[df1['freq']==4].Reference)),1000)
s4 = rand_smpl(list(set(df1[df1['freq']==5].Reference)),1000)
s5 = rand_smpl(list(set(df1[df1['freq']==6].Reference)),1000)
s6 = rand_smpl(list(set(df1[df1['freq']==7].Reference)),1000)
s7 = rand_smpl(list(set(df1[df1['freq']==8].Reference)),1000)
s8 = rand_smpl(list(set(df1[df1['freq']>=9].Reference)),1000)


#From df1 select first data sentences corresponding to references and remove from df1
#import multiprocessing
def validation_Indices(j):
    try:
        mylist = [list(df1[df1.Reference==i].index)[0] for i in j]
    except:
        print('error with item')
    return mylist

executor = concurrent.futures.ThreadPoolExecutor(12)
futures = [executor.submit(validation_Indices, item) for item in [s1,s2,s3,s4,s5,s6,s7,s8]]
concurrent.futures.wait(futures)

validationIndices=[]
for i in futures:
    validationIndices.append(i.result())

validationIndices = [item for sublist in validationIndices for item in sublist]

# validationIndices = [] 
# for j in [s1,s2,s3,s4,s5,s6,s7,s8]:
#     tmp = [list(df1[df1.Reference==i].index)[0] for i in j]
#     validationIndices.append(tmp)
#Save Validation set
dfValidate = df1.ix[validationIndices]
dfValidate['freq'] = dfValidate.freq-1
#Drop Validation Set from data frame
dfTrain= df1.drop(df1.index[validationIndices])


dfValidate.to_csv('/home/lanna/Dropbox/Insight/X-Cite/dfValidate.csv')
dfTrain.to_csv('/home/lanna/Dropbox/Insight/X-Cite/dfTrain.csv')
#################################################
# TRAINING SET!
#################################################
#dfTrain = df1.copy()
#Multiple labels for each sentence
#What are the unique sentences?
sentences2 = list(set(dfTrain['Sentence']))

#LabeledSentence objects:
#Each such object represents a single sentence, and consists of two simple lists: a list of words and a list of labels
def label_Sentences(sentence):
    #Create a LabeledSentence object:
    #1. For each unique sentences (first argument)-- tokenized sentence (split sentences by words) 
    #2. The corresponding Reference labels (second argument)
    tmp = dfTrain.ix[dfTrain[dfTrain['Sentence']==sentence].index.tolist()].Reference.tolist()
    return tmp

executor = concurrent.futures.ProcessPoolExecutor(12)
futures = []
num = 0
for item in sentences2:
    num += 1
    print(str(round(num*100/len(sentences2),3))+'%', end='\r')
    futures.append(executor.submit(label_Sentences, item))
concurrent.futures.wait(futures)

tags=[]
for i in futures:
    tags.append(i.result())

#Pickle/save list of tags for sentences
f = open('/home/lanna/Dropbox/Insight/tags','wb')
pickle.dump(tags,f)    

LabeledSentences = []
for i in range(0,len(sentences2)):
    LabeledSentences.append(doc2vec.LabeledSentence(sentences2[i].split(), tags[i]))
  
#https://linanqiu.github.io/2015/05/20/word2vec-sentiment/

nfeatures = 300
model = gensim.models.doc2vec.Doc2Vec(workers = 10, size=nfeatures, window=10, min_count=1,alpha=0.025, min_alpha=0.025)
#Build the vocabulary table: digesting all the words and filtering out the unique words, and doing some basic counts on them
model.build_vocab(LabeledSentences) 

#Train Doc2Vec
from random import shuffle
#Randomize order of sentences
for epoch in range(10):
    shuffle(LabeledSentences)
    model.train(LabeledSentences)
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

# store the model to mmap-able files
model.save('/home/lanna/Dropbox/Insight/X-Cite/doc2vec_model.doc2vec')

#################################################
# VALIDATION SET
#################################################
#http://blog.dato.com/how-to-evaluate-machine-learning-models-part-2a-classification-metrics
model_loaded = gensim.models.doc2vec.Doc2Vec.load('/home/lanna/Dropbox/Insight/X-Cite/doc2vec_model.doc2vec')
dfValidate = pd.read_csv('/home/lanna/Dropbox/Insight/X-Cite/dfValidate.csv')

def cleanInput(userInput):
    Input1 = re.sub("[^a-zA-Z]", " ", userInput) #Only extract words
    Input = Input1.lower().split() #Tokenize sentences: convert to lower case and split them into individual words
    return Input

#Function to determine rank of 
def rankValidation(datf):
    sentenceInput = datf['Sentence']
    actualReference = datf['Reference']
    freq = datf['freq']
    cI = cleanInput(sentenceInput)  #Clean userInput sentence
    userVec = model_loaded.infer_vector(cI) #Infer a vector for given post-bulk training document. Document should be a list of (word) tokens.
    output = model_loaded.docvecs.most_similar(positive=[userVec], topn = 60921600) #Find the top-N most similar docvecs known from training. 
    tt = pd.DataFrame(output, columns=['Citation','probability'])   #turn into pandas dataframe
    refIndex = tt[tt.Citation==actualReference].index.tolist()[0]    #Find index where actualReference occurs
    rank = tt.rank(0).probability.iloc[refIndex]
    prob = tt.probability[refIndex]
    out = pd.DataFrame(data = {'Reference': [actualReference], 'Sentence': [sentenceInput], 'freq': [freq], 'model_prob': [prob], 'model_rank': [rank]})
    return out

validationResults = pd.DataFrame(columns=['Reference','Sentence','freq','model_prob','model_rank'])
validationResults.to_csv('/home/lanna/Dropbox/Insight/X-Cite/validationResults.csv',header=True, index=False)

chunk = []
for i in range(0,len(dfValidate)+100,100):
    chunk.append(i)

#BREAK THIS UP SOMEHOW BY GROUPS OF 10!
n = 0
while n < 8000:
    dftmp = dfValidate.copy().iloc[range(chunk[n],chunk[n+1])]
    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = []
    num = 0
    for index, row in dftmp.iterrows():
        num+=1
        print(str(round(num*100/len(dftmp),3))+'%', end='\r')
        futures.append(executor.submit(rankValidation, row))
    concurrent.futures.wait(futures)

    for j in futures:
        tmp = j.result()
        with open('/home/lanna/Dropbox/Insight/X-Cite/validationResults.csv',mode='a') as f:
            tmp.to_csv(f, header=False, index=False)
    n+=1

#Read in Results
vResults = pd.read_csv('/home/lanna/Dropbox/Insight/X-Cite/validationResults.csv')
vResults['freq'] = vResults.freq-1

vAvgs = pd.DataFrame(columns=['freq','mean_model_prob'])
for i in range(1,9):
    mean_model_prob = vResults[vResults.freq==i].model_prob.mean()
    vAvgs = vAvgs.append(pd.DataFrame(data = {'freq':[i],'mean_model_prob':[mean_model_prob]}))

#Now for if frequency is 9 or greater
mean_model_prob = vResults[vResults.freq>8].model_prob.mean()
vAvgs = vAvgs.append(pd.DataFrame(data = {'freq':9,'mean_model_prob':[mean_model_prob]}))

import matplotlib.pyplot as plt
plt.plot(vAvgs.freq, vAvgs.mean_model_prob)
plt.ylabel('Mean Cosine similarity')
plt.xlabel('Citation Frequency')
plt.show()

#################################################
# PREDICT
#################################################
# load the model back
model_loaded = gensim.models.doc2vec.Doc2Vec.load('/home/lanna/Dropbox/Insight/X-Cite/doc2vec_model.doc2vec')

userInput = 'Chimpanzee habituation allows for direct monitoring and hence precise censuses, but is a lengthy process which is necessarily restricted to small numbers of individuals, and may not be ethically appropriate or logistically feasible for many populations'  #Input sentence

def cleanInput(userInput):
    Input1 = re.sub("[^a-zA-Z]", " ", userInput) #Only extract words
    Input = Input1.lower().split() #Tokenize sentences: convert to lower case and split them into individual words
    return Input

cI = cleanInput(userInput)  #Clean userInput sentence

#Infer a vector for given post-bulk training document. Document should be a list of (word) tokens.
userVec = model_loaded.infer_vector(cI) 
#Find the top-N most similar docvecs known from training. Positive docs contribute positively towards the similarity, negative docs negatively. 
#This method computes cosine similarity between a simple mean of the projection weight vectors of the given docs. Docs may be specified as vectors, integer indexes of trained docvecs, or if the documents were originally presented with string tags, by the corresponding tags.
#Here, doc is given as infered vector
output = model_loaded.docvecs.most_similar(positive=[userVec])
the_result = pd.DataFrame(output, columns=['Citation','probability'])

