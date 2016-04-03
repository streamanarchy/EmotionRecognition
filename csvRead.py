import csv
import re
from neo4jrestclient.client import GraphDatabase
from neo4jrestclient import client
import glob
#from py2neo import neo4j, node, rel
#gdb = neo4j.GraphDatabaseService()
eValue = 0
gdb = GraphDatabase("http://localhost:7474","neo4j","this is me")

def csvRead(filename):
	count = 0
	relationwords = gdb.labels.create("relationwords")

	with open(filename,'rb') as csvfile:

		reader = csv.reader(csvfile , delimiter= ',',quotechar = "\"")
		for row in reader:
			if count <= 10200:
				continue
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
			nodelist = []
			wordNode = gdb.node()
			for word in wordBuffer:
				q = "start n = node(*) match(n:relationwords{word:\'"+word+"\'}) return n"
				result = gdb.query(q,returns=(client.Node))

				if result.__len__() == 1:
					if eValue == '0':
						result[0][0].set("neutral",result[0][0].get("neutral")+1)
					if eValue == '2':
						result[0][0].set("positive",result[0][0].get("positive")+1)
					if eValue == '4':
						result[0][0].set("negative",result[0][0].get("negative")+1)
					nodelist.append(result[0][0])
				elif result.__len__() == 0:
					if eValue == '0':
						wordNode = gdb.node.create(word = word, positive = 0, negative=0, neutral = 1)
						nodelist.append(wordNode)
						relationwords.add(wordNode)
					if eValue == '2':
						wordNode = gdb.node.create(word = word, positive = 1, negative=0, neutral = 0)
						nodelist.append(wordNode)
						relationwords.add(wordNode)
					if eValue == '4':
						wordNode = gdb.node.create(word = word, positive = 0, negative=1, neutral = 0)
						nodelist.append(wordNode)
						relationwords.add(wordNode)

			for x in xrange(0,nodelist.__len__()-1):
				rellist=nodelist[x].relationships.outgoing()[:]
				relendnode =[]
				for eachrel in rellist:
					relendnode.append(eachrel.end)
				if rellist != []:
					if nodelist[x+1] not in relendnode:
						nodelist[x]._used_with(nodelist[x+1])
				else:
					nodelist[x]._used_with(nodelist[x+1])



			count = count+1
			if count%50 == 0:
				print count


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

	filelist = glob.glob("/home/project/Documents/Project/textdata/second*")
	for file in filelist:
		csvRead(file)
		print file," completed"
