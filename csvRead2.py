import csv
from py2neo import neo4j, node
from neo4jrestclient.client import GraphDatabase

#auth =neo4j.authenticate("127.0.0.1:7474",'neo4j','this is me')
gdb = GraphDatabase("http://localhost:7474","neo4j","this is me")


eValue = 0
def csvRead():
	count = 0
	word = gdb.labels.create("word")
	with open('/home/project/Documents/Project/ratings.csv','rb') as csvfile:
		
		reader = csv.reader(csvfile , delimiter= ',',quotechar = "\"")
		for row in reader:
			wordNode = gdb.nodes.create(word = row[1],VLow = row[2],VHigh = row[3] , ALow = row[4], AHigh = row[5], DLow = row[6], DHigh = row[7])
			word.add(wordNode)
			count =count +1
			if count%100 == 0:
				print count
			if count == 1000:
				break



if __name__ == "__main__":
	csvRead()
