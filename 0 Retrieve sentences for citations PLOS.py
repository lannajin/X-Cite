from bs4 import BeautifulSoup, Tag
from urllib.request import urlopen
import urllib
import json
import re
import pandas as pd
import concurrent.futures

#Scrape PLOS for journal URL and ISSN
journalUrls = ['http://www.plosbiology.org',
	 'http://www.plosgenetics.org',
	'http://www.ploscompbiol.org',	#
	'http://www.plosmedicine.org',
	'http://www.plosone.org',
	'http://www.plosntds.org',
	'http://www.plospathogens.org']

ISSN = ["1545-7885","1553-7404", "1553-7358", "1549-1676", "1932-6203", "1935-2735", "1553-7374"]
#Create a Data frame to store the URL and ISSN of each journal in PLOS
df0 = pd.DataFrame(data={'URL':journalUrls,'ISSN': ISSN})

#Through the CrossRef API, extract works from each PLOS journal by their ISSN for years 2003-2016
def getJournals(issn, url):
	DOIs = []
	for y0 in range(2003,2016):
		response = urlopen("http://api.crossref.org/journals/"+issn+"/works?filter=from-pub-date:"+str(y0)+"-01-01,until-pub-date:"+str(y0+1)+"-01-01,type:journal-article&rows=1000").read().decode('utf-8')
		responseJson = json.loads(response)
		for i in range(0,len(responseJson.get("message").get("items"))):
			if responseJson.get("message").get("items")[i].get("type") == 'journal-article':
				DOIs.append(responseJson.get("message").get("items")[i].get("DOI"))

	out = pd.DataFrame(data={'URL':[url]*len(DOIs),'DOI': DOIs})
	return out

executor = concurrent.futures.ProcessPoolExecutor(12)
futures = []
num = 0
for index, row in df0.iterrows():
    num += 1
    print(str(round(num*100/len(df0),3))+'%', end='\r')
    futures.append(executor.submit(getJournals, row['ISSN'], row['URL']))
concurrent.futures.wait(futures)

jDOIs=pd.DataFrame()
for i in futures:
    jDOIs = jDOIs.append(i.result())


jDOIs.to_csv('/home/lanna/Dropbox/Insight/jDOIs_PLOS', index=False)
########################################################################################################################
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.request import urlopen
import re
import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
import sklearn
import numpy as np
import pandas as pd
import itertools
import concurrent.futures
from itertools import groupby

def getCitations(url):
	try:
		try:
			html = urlopen(url)
		except:
			return
		bsObj = BeautifulSoup(html.read())

		if not bsObj.findAll("p",{"class","type-article"}):
			return #pd.DataFrame(columns=['Reference','Sentence', 'DOI'])

		refs = [i.next_sibling for i in bsObj.ol.findAll("a",{"class","link-target"})]

		dois = []
		for i in bsObj.findAll("ol",{"class","references"})[0]:
			try:
				dois.append(i.ul.get('data-doi'))
			except:
				dois.append("")

		article = bsObj.find("div",{"class","article-text"})

		#Remove any text from figures 
		#Remove any Figures  
		for tag in article.findAll("div",{"class","figure"}):
		    tag.replaceWith('')

		 #Remove supporting info
		for tag in article.findAll("div",{"class","supplementary-material"}):
		    tag.replaceWith('')

		#Function to check if numeric or not
		def checkInt(str):
			try:
				int(str)
				return True
			except:
				return False

		#Print list of Citation numbers. Will be indexed as "CR"+number
		refids = [i.get('id') for i in article.ol.findAll("a",{"class","link-target"})]

		citeNum = []
		for name in article.findAll("a",{"class","ref-tip"}):
			if checkInt(name.get_text()):	#If ref-tip is numeric, e.g., '2'
				if int(name.get_text()) <= len(refs):
					citeNum.append(int(name.get_text()))
				else:	#For when the citations refers to a year
					refurl = name.get('href')	#Get URL for the particular reference
					refurl = re.sub(r"\#","",refurl)	#get rid of #
					citeNum.append(refids.index(refurl)+1)
			else:	#When accoiated with name instead, e.g., "Lawton et al."
				refurl = name.get('href')	#Get URL for the particular reference
				refurl = re.sub(r"\#","",refurl)	#get rid of #
				citeNum.append(refids.index(refurl)+1)

		#Find all instances of when class is ref-tip
		refTags = article.findAll("a",{"class","ref-tip"})

		def findSent(firstElement, nextATag):
			text = []
			firstElement = firstElement.next
			while firstElement != nextATag:
				firstElement = firstElement.next
				if str(type(firstElement)) != "<class 'bs4.element.Tag'>":
					text += firstElement.string
			text = "".join(text)
			return text

		#First sentence
		firstSent = str(article.findAll("a",{"class","ref-tip"})[0].findPreviousSibling(text=True))
		
		test = []
		test.append(firstSent)	#Add first sentence to list of sentences (test)

		#Add bulk of sentences to list of sentences (test)
		for i in range(0,len(refTags)-2):
			test.append(findSent(firstElement = refTags[i], nextATag = refTags[i+1]))

		#Add last sentence to list of sentences
		lastSent = [findSent(firstElement = refTags[len(refTags)-2], nextATag = refTags[len(refTags)-1])]
		lastlast = article.findAll("a",{"class","ref-tip"})[len(refTags)-1].findNextSibling(text=True)
		lastSent.append(sent_detector.tokenize(lastlast.strip())[0])
		lastSent = "".join(lastSent)

		#Add Last sentence
		test.append(lastSent)

		#Clean up sentences
		sent=[]
		for i in range(0,len(test)):
			tt = BeautifulSoup(test[i]).get_text()
			tt = tt.replace("\n", "")
			sent.append(re.sub('(?<=\.)(?=[a-zA-Z])', ' ', tt))

		#Deal with sentences that may be broken up by citations.
		Xs = sent.copy()

		def addsent(i):
			k = 1
			period = False
			while period == False:
				if len(range(0,len(sent_detector.tokenize(Xs[i+k].strip())))) !=0:
					for j in range(0,len(sent_detector.tokenize(Xs[i+k].strip()))):
						if sent_detector.tokenize(Xs[i+k].strip())[j][-1] ==".":	#if the last character in the sentence is a period
							period = True
							break
						else: 
							k += 1
				else:
					k+=1
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

		#join words back together into string
		sentences2 = []
		for i in words:
			sentences2.append(' '.join(i))

		#Find citations with multiple sentences and find average
		df1 = pd.DataFrame(data = {'citeNum': list(citeNum2), 'Sentence': sentences2})
		df2 = pd.DataFrame(data = {'citeNum': range(1,len(refs)+1), 'Reference': list(refs), 'DOI': dois})
		dat = pd.merge(df1, df2, on = "citeNum")
		out = dat[['Reference','Sentence','DOI']]
		return out
	except:
		return

#############################
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

jDOIs = pd.read_csv("/home/lanna/Dropbox/Insight/jDOIs_PLOS")

##
df2 = pd.DataFrame(columns=['Reference','Sentence', 'DOI'])
df2.to_csv('/home/lanna/Dropbox/Insight/df_PLOS.csv',header=True, index=False)

chunk = []
for i in range(0,len(jDOIs)+10,10):
    chunk.append(i)

chunk[-1] = len(jDOIs)	#Since not evenly divided by 100, make last bit up to length of jDOIs

import time
from functools import wraps
n = 0
num = 0
start = time.time()
while n < len(jDOIs):
	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	num+=1
	print(str(round(num*100/len(chunk),3))+"%; Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds), end='\r')
	dftmp = jDOIs.copy().iloc[range(chunk[n],chunk[n+1])]
	executor = concurrent.futures.ProcessPoolExecutor(11)
	futures = []
	for index, row in dftmp.iterrows():
		try:
			f1 = executor.submit(getCitations, "http://journals.plos.org/"+re.findall(r"(?<=http://www\.).*(?=\.org)",row['URL'])[0]+"/article?id="+row['DOI'])
			futures.append(f1)
		except:
			pass
		#futures.append(executor.submit(getCitations, "http://journals.plos.org/"+re.findall(r"(?<=http://www\.).*(?=\.org)",row['URL'])[0]+"/article?id="+row['DOI']))
	concurrent.futures.wait(futures)

	for j in futures:
		try:
			tmp = j.result()
			with open('/home/lanna/Dropbox/Insight/df_PLOS.csv',mode='a') as f:
				tmp.to_csv(f, header=False, index=False)
		except:
			pass
	n+=1

