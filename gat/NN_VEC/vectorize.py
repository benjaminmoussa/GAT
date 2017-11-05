import pandas as pd
from pandas import Series, DataFrame
import re
import string
from collections import Counter
def makeBag(path):
    data = pd.read_csv(path)
    data.columns = ['code', 'move', 'domain', 'type', 'resources', 'key_words']
    move = data['move'].str.lower()
    key_words = data['key_words'].str.lower()

    word_set = set()
    ########og cameo
    for x in move:
        x= re.sub(r"[\(\[].*?[\)\]]", "", x)
        x = x.replace(",", '')
        x = x.replace('"', '')
        x = x.replace('/', ' ')
        tempwords = x.split(' ')
        for y in tempwords:
            word_set.add(y)
            #print(y)
    for x in key_words:
        #for blanks
        if type(x) == float:
            continue
        x = re.sub(r"[\(\[].*?[\)\]]", "", x)
        x = x.replace(",",'')
        x = x.replace('"', '')
        x = x.replace('/', ' ')
        tempwords = x.split(' ')
        for y in tempwords:
            word_set.add(y)
            #print(y)
    word_list = list(word_set)
    return word_list
def vectorizeTextFile(bag):
    #read text file from filepath
    file = open(filepath, "r")
    text = file.read()
    #replace any newlines with no newlines so that we have contigous paragraph, just in case paragraph is split into random lines (arbitrary text file case not formated)
    text = text.replace("\n", "")
    #get rid of trailing whitespaces
    text = text.rstrip()
    #close file
    file.close()
    #use regular expression to split paragraph into strings, takes into account things like i.e. and Mr.
    regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    #split using regex, now we have a list of sentences
    sentences = re.split(regex, text)
    #create set
    w_set = set()
    pair_list = []
    for x in range(len(sentences)):
        words = sentences[x].split(" ")
        vector = createVector(bag, words)
        pair_list.append([vector, sentences[x]])
    df_vectors = DataFrame(pair_list, columns = ['vector', 'sentence'])
    return df_vectors
def vectorizeCSV(bag, filepath):
    examples = pd.read_csv(filepath)
    examples.columns = ['c_number','c_event', 'ex']
    examples['ex'] = examples['ex'].apply(lambda x:createWord_List(x))
    examples['ex'] = examples['ex'].apply(lambda x:createVector(bag,x))
    examples['c_event'] = examples['c_event'].apply(lambda x:createWord_List(x))
    examples['c_event'] = examples['c_event'].apply(lambda x:createVector(bag,x))
    return examples
def createWord_List(string):
    string = re.sub(r"[\(\[].*?[\)\]]", "", string)
    string = string.lower()
    string = string.replace("\n", "")
    string = string.rstrip()
    string = string.replace(",",'')
    string = string.replace('"', '')
    string= string.replace('/', '')
    string = string.replace('-',' ')
    string = string.replace('  ', ' ')
    words = string.split(" ")
    #regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    #string = re.split(regex, string)
    #print(string)
    return string

def createVector(bag, word_list):
    vector = []
    #create vector
    for x in bag:
        vector.append(0)
        for y in word_list:
            if x == y:
                vector.pop()
                vector.append(1)
                break
    if len(vector) != 1167:
        print('meow')
    return vector
#def labelData(path,bag)
#create bag of words############################
path = '/Users/Moussa/documents/cameo.csv'
filepath = '/Users/Moussa/documents/All_Samples.csv'
bag = makeBag(path)
bag = Series(bag)
bag.to_csv('/Users/Moussa/documents/BagOfWords.csv', index = False)
#################################################
#bag = pd.read_csv('/Users/Moussa/documents/BagOfWords.csv')
#make bag of words dataframe to list
bag = Series(bag).tolist()
#vector = vectorizeTextFile(bag, "/Users/Moussa/Documents/L33T/text2.txt")
vector = vectorizeCSV(bag,filepath)
vector.to_pickle('/Users/Moussa/documents/AllSampleVectors.csv')
