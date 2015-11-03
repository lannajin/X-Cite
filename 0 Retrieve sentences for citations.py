from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib
import json
import re
import pandas as pd
import pickle

#Scrape BioMed Central's for journal URL and ISSN
html = urlopen("http://beta.biomedcentral.com/journals/all-journals")
bsObj = BeautifulSoup(html.read())
stack = bsObj.findAll("ol",{"class":"ListStack"})
jurl = []
for text in stack:
	for links in text.findAll("a"):
		jurl.append(links.get('href'))

#Create a Data frame to store the URL and ISSN of each journal in BioMed Central
df0 = pd.DataFrame()
for url in jurl:
	try:
		html = urlopen(url)
		bsObj = BeautifulSoup(html.read())
		issn = bsObj.find("dd").get_text()
		df0 = df0.append(pd.DataFrame(data = {'URL': [url], 'ISSN': [issn]}))
	except AttributeError:
		pass
	except urllib.error.URLError:
		pass
	except NameError:
		pass

#Remove entries in df0 that have weird ISSNs
mask = df0['ISSN'].str.len() < 10
df0 = df0.loc[mask]

#Through the CrossRef API, extract all the works from BMC Ecology by their ISSN
#Max number of rows allowed = 1000
def getJournals(issn):
	response = urlopen("http://api.crossref.org/journals/"+issn+"/works?sample=1000").read().decode('utf-8')
	responseJson = json.loads(response)
	DOIs = []
	for i in range(0,len(responseJson.get("message").get("items"))):
		DOIs.append(responseJson.get("message").get("items")[i].get("DOI"))
	return DOIs

jDOIs = pd.DataFrame()
for index, row in df0.iterrows():
	try:
		print(row['ISSN'])
		doi = getJournals(row['ISSN'])
		jDOIs = jDOIs.append(pd.DataFrame(data={'URL':[row['URL']]*len(doi),'DOI':doi})) #Retrieve dois from the journals
	except:
		pass

jDOIs.to_pickle("jDOIs")
########################################################################################################################
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
import sklearn
import numpy as np
import pandas as pd
import pickle
import itertools

def getCitations(url, doi):
	html = urlopen(url+doi)
	bsObj = BeautifulSoup(html.read())

	refs = []
	for i in bsObj.ol.findAll("cite",{"class","CitationContent"}):
		refs.append(i.get_text())

	#Print list of Citation numbers. Will be indexed as "CR"+number
	citeNum = []
	for name in bsObj.findAll("span",{"class","CitationRef"}):
		citeNum.append(int(name.get_text()))

	test = re.split(r"(?:<span class=\"CitationRef\"><a href=\"#CR\d+\">\d+<\/a><\/span>)", str(bsObj))
	test[0] = re.findall(r"(?!.*>).*$",test[0])[0]	#Clean up first sentences' citation by getting rid of surrouding HTML
	test.pop(len(test)-1)	#Get rid of HTML after last bit

	sent=[]
	for i in range(0,len(test)):
		tt = BeautifulSoup(test[i]).get_text()
		tt = tt.replace("\n", "")
		sent.append(re.sub('(?<=\.)(?=[a-zA-Z])', ' ', tt))

	#sent.append(re.sub("[^a-zA-Z/.,–]", " ", i.previousSibling))

	#Deal with sentences that may be broken up by citations.
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	Xs = sent.copy()

	def addsent(i):
		k = 1
		period = False
		while period == False:
			for j in range(0,len(sent_detector.tokenize(Xs[i+k].strip()))):
				if sent_detector.tokenize(Xs[i+k].strip())[j][-1] ==".":	#if the last character in the sentence is a period
					period = True
					break
				else: 
					k += 1
		return([k,j])

	i=0
	while i < (len(Xs)-1):
		tt = addsent(i)
		k = tt[0]
		j = tt[1]
		new = [sent_detector.tokenize(Xs[i].strip())[-1]]
		x = 1
		while x < k:
			new.append(Xs[i+x])
			x += 1
		
		new.append("".join(sent_detector.tokenize(Xs[i+x].strip())[0:j+1]))
		new = "".join(new)
		for m in range(0,k):
		#Add sentence to other sentences, unless it is a sentence with multiple citation
			if (not re.findall(r"(^ *,{1} *)$",sent[i+m])) and (not re.findall(r"(^ *–{1} *)$",sent[i+m])):
				Xs[i+m] = new
		i+=k

	#Clean last sentences too
	if Xs[-1] != Xs[-2]:
		Xs[-1] = sent_detector.tokenize(Xs[-1].strip())[-1]

	#Deal with multiple citations either separated by commas, or hyphens
	commas = [i for i,x in enumerate(Xs) if not not re.findall(r"(^ *,{1} *)$",x)]
	multcites = [i for i,x in enumerate(Xs) if not not re.findall(r"(^ *–{1} *)$",x)]

	#If there's a comma, add the sentence before to the blank sentence
	for i in commas:
		Xs[i] = Xs[i-1]
		#If comma is "–", find first instances of a sentences that isn't "–"
		if not not re.findall(r"(^ *–{1} *)$",Xs[i]):
			j = 2
			while not not re.findall(r"(^ *–{1} *)$",Xs[i]):
				Xs[i] = Xs[i-j]
				j += 1

	#For citations with hyphens
	for i in multcites:	#e.g., for 4-8
		Xs[i] = Xs[i-1]	#Assign sentences of previous index to index, e.g., sentence 4 to sentence 8
		for j in range(1,citeNum[i]-citeNum[i-1]):	#For the citations inbetween, e.g., 5,6,7
			Xs.append(Xs[i-1])	#Add sentence 4 to the end for citations 5,6 and 7
			citeNum.append(citeNum[i-1]+j)	#Add citation reference number for 5,6,7

	sent2 = []
	for i in Xs:
		sent2.append(re.sub("[^a-zA-Z]", " ", i))

	#Remove sentences that are blank
	blanks = [i for i,x in enumerate(sent2) if not not re.findall(r"^\s*\s$",x)]   #Identify blank citation sentences
	sentences = [i for j, i in enumerate(sent2) if j not in blanks]    #Remove blank citation sentences from cleaned sentences
	citeNum2 = [i for j, i in enumerate(citeNum) if j not in blanks] #Remove blank citations from reference numbers

	#Tokenize sentences: convert to lower case and split them into individual words
	words = []
	for i in sentences:
		words.append(i.lower().split())

	# Remove stop words from "words". Do not need this for vectorization
	#words = [w for w in words if not w in stopwords.words("english")]

	#join words back together into string
	sentences2 = []
	for i in words:
		sentences2.append(' '.join(i))

	#Find citations with multiple sentences and find average
	df1 = pd.DataFrame(data = {'citeNum': list(citeNum2), 'Sentence': sentences2})
	df2 = pd.DataFrame(data = {'citeNum': range(1,len(refs)+1), 'Reference': list(refs)})
	dat = pd.merge(df1, df2, on = "citeNum")
	out = dat[['Reference','Sentence']]
	return out

#############################
jDOIs = pd.read_pickle("jDOIs")

df1 = pd.DataFrame(columns=['Reference','Sentence'])
df1.to_csv('/home/lanna/Dropbox/Insight/df1.csv',header=True, index=False)
num = 0
for index, row in jDOIs.iterrows():
	num+=1
	print(str(round(num*100/len(jDOIs),3))+'%', end='\r')
	try:
		with open('/home/lanna/Dropbox/Insight/df1.csv',mode='a') as f:
			getCitations(url = row['URL']+"articles/", doi = row['DOI']).to_csv(f, header=False, index=False) #Retrieve dois from the journals
	except:
		print(row['DOI'])
		pass

# df1.to_pickle("df1")
# df1.to_csv("/home/lanna/Dropbox/Insight/df1.csv",header=True)


# df1 = pd.DataFrame()
# num = 0
# for index, row in jDOIs.iterrows():
# 	num+=1
# 	print(str(round(num*100/len(jDOIs),3))+'%', end='\r')
# 	try:
# 		df1 = df1.append(getCitations(url = row['URL']+"articles/", doi = row['DOI'])) #Retrieve dois from the journals
# 	except:
# 		pass
