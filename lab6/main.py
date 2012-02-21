#!/usr/bin/env python2
import numpy as np

def main():
	tspfile = open('tsp.txt', 'r')
	tsp = []
	S = []
	for line in tspfile:
		tsp.append(line[:-2].split("\t"))
	S = [[float(y) for y in x] for x in tsp]

	N = [2,3,7,9]
	t = np.ones( (len(S),len(S)) )
	#print np.add(generate_tour(2,S,N,t,2,2),1)
	for i in range(0,50):
		print ant_colony_optimization(S,100,20,1.3,2.5,0.5,0.5,0.001)
	

def next_state(s,S,N,t,alfa,beta):
	P = np.zeros(len(N))
	for i in range(0,len(N)):
		s2 = N[i]
		d = distance(S[s][0],S[s][1],S[s2][0],S[s2][1])
		P[i] = t[s][s2]**alfa * (1/d)**beta
	P = np.divide(P,sum(P))
	for i in range(1,len(P)-1):
		P[i] = P[i] + P[i-1]
	P[-1] = 1
	r = np.random.random()
	num = 0
	while r > P[num]:
		num = num + 1
	statenext = N[num]
	return statenext

def distance(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def total_distance(citys,trip):
	length = 0
	for r in range(0,len(trip)-1):
		length = length + distance(citys[trip[r]][0],citys[trip[r]][1],citys[trip[r+1]][0],citys[trip[r+1]][1])
	return length

def generate_tour(startstate,S,N,t,alfa,beta):
	N.remove(startstate)
	episode = [startstate]
	s = startstate
	while N != []:
		s2 = next_state(s,S,N,t,alfa,beta)
		episode.append(s2)
		N.remove(s2)
		s = s2
	episode.append(startstate)
	return episode

def ant_colony_optimization(S,maxit,popsize,alfa,beta,rho,Q,tau0):
	tau = []
	for i in range(0,len(S)):
		tau.append([])
		for j in range(0,len(S)):
			tau[i].append(0.001)
	N = list(np.arange(len(S)))
	T = list(np.arange(len(S)))
	for it in range(0,maxit):
		for p in range(0,popsize):
			s0 = np.random.randint(0,len(N))
			T[p] = generate_tour(s0,S,N[:],tau,alfa,beta)
		for s in range(0,len(S)):
			for s2 in range(0,len(S)):
				tau[s][s2] = (1 - rho) * tau[s][s2]
		for p in range(0,popsize):
			for n in range(0,len(T[p])-1):
				s = T[p][n]
				s2 = T[p][n+1]
				tau[s][s2] = tau[s][s2] + Q / total_distance(S,T[p])
	L = []
	for tour in T:
		L.append(total_distance(S,tour))
	best_tour = min(L)
	tid = np.argmin(L)
	return best_tour, np.add(T[tid],1)

		

if __name__ == "__main__":
   main()