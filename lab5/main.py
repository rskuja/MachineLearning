#!/usr/bin/env python2
import numpy as np

D = []
E = []

def main(ex = [1,1,1,3]):
	dfile = open('d.txt', 'r')
	for line in dfile:
		D.append(int(line[:-2]))

	xfile = open('x.txt', 'r')
	for line in xfile:
		E.append(line[:-2].split("\t"))

	print naive_bayes(ex)
	print test_naive_bayes()

def naive_bayes(ex):
	values = np.unique(D)
	prob = []
	for currdec in range(0,len(values)):
		C1 = float(D.count(values[currdec]))
		C2 = len(D)
		prob.append(C1/C2)
		for currattr in range(0,len(E[0])):
			C3 = 0
			for i in range(0,len(E)):
				if int(E[i][currattr]) == int(ex[currattr]) and D[i] == values[currdec]:
					C3 = C3 + 1
			prob[currdec] = prob[currdec] * (C3 / C1)	
	decision = np.argmax(prob) + 1
	return decision

def test_naive_bayes():
	for ex in range(0,len(E)):
		rez = naive_bayes(E[ex])
		print D[ex],rez

if __name__ == "__main__":
   main()