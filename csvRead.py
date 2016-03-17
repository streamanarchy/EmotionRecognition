import csv
import re
from neo4jrestclient.client import GraphDatabase
from py2neo import neo4j, node, rel
#gdb = neo4j.GraphDatabaseService()
eValue = 0
gdb = GraphDatabase("http://localhost:7474","neo4j","this is me")

def csvRead():

	relationwords = gdb.labels.create("relationwords")

	with open('/home/project/Documents/Project/words.csv','rb') as csvfile:
		
		reader = csv.reader(csvfile , delimiter= ',',quotechar = "\"")
		for row in reader:
			eValue = row[0]
			wordBuffer = row[-1].split()
			for eachWord in reversed(wordBuffer):

				if eachWord[0] == '@' or eachWord[0] == '#':
					wordBuffer.remove(eachWord)
					
				elif eachWord[:4] == "http" or eachWord[:3] == "www":
					wordBuffer.remove(eachWord)



			cleanTextWithSymbols = '  '.join(wordBuffer)
			cleanText = cleanTextWithSymbols.translate(None,",.!?*&;:'(){}[]").lower()
			wordBuffer = re.findall(r"[\w']{4,}",cleanText)

			for word in wordBuffer:
				wordNode = gdb.node.create(word = word, positive = 0, negative=0, neutral = 0)
				relationwords.add(wordNode)


			"""for eachWord in wordBuffer:
				if eachWord[0] == '@' or eachWord[0] == '#':
					continue
				elif eachWord.__len__() <=3:
					continue
				if eachWord[:4] == "http" or eachWord[:3] == "www":
					continue
				if eachWord[-3] == '...':
					eachWord = eachWord[:-3]
				if eachWord[-1] in ['?','\,',',','.','!']:	
					eachWord = eachWord[:-1]
							
				if eachWord[0] in ['?','\,',',','.','!']:	
					eachWord = eachWord[1:]
				print eachWord"""

if __name__ == "__main__":
	csvRead()
