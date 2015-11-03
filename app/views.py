from flask import render_template, request
from app import app
import re
import pandas as pd
import numpy as np
import gensim
from gensim.models import doc2vec

model_loaded = gensim.models.doc2vec.Doc2Vec.load('doc2vec_model.doc2vec')

def cleanInput(userInput):
    Input1 = re.sub("[^a-zA-Z]", " ", userInput) #Only extract words
    Input = Input1.lower().split() #Tokenize sentences: convert to lower case and split them into individual words
    return Input

@app.route('/')
@app.route('/index')
def index():
	return render_template("input.html")
	
@app.route('/input')
def cities_input():	#AKA input_sentence
	return render_template("input.html")
	
@app.route('/output')
def cities_output():
	var = request.args.get('ID')
	num = request.args.get('num')
	cI = cleanInput(var) #Clean userInput sentence
	#Infer a vector for given post-bulk training document. Document should be a list of (word) tokens.
	userVec = model_loaded.infer_vector(cI) 
	output = model_loaded.docvecs.most_similar(positive=[userVec], topn=int(num))
	the_result = pd.DataFrame(output, columns=['Citation','probability'])
	the_result['probability'] = [round(probs*100,1) for probs in the_result.probability]
	return render_template("output.html", citations=the_result, sentence = var, num=num)
