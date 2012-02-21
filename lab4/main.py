#!/usr/bin/env python2
import numpy as np
import math

def main():
   #print solution_to_bits([0.1875,-1.5469,0.9219],8)
   #print bits_to_solution([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],3,8)
   
   X = [[0,0],[0,1],[1,0],[1,1]]
   #W = [[1.5,-0.5,-0.5],[2,-1,-1],[1.5,-0.25,-0.5],[-0.3125,-0.1406,-0.4219],[0.0313,0.375,0.0938]]
   D = [1,1,1,0]
   #print error_function(X,W,D)
   
   # efile = open('e.txt', 'r')
   # e = []
   # for line in efile:
   #    e.append(float(line[:-2]))
   # print compute_probabilities(e)

   # wfile = open('w.txt', 'r')
   # w = []
   # for line in wfile:
   #    w.append(line[:-2].split("\t"))
   # w = [[float(y) for y in x] for x in w]

   # pfile = open('p.txt', 'r')
   # p = []
   # for line in pfile:
   #    p.append(float(line[:-2]))

   # print selection(w,p,2)  

   # print single_point_crossover(w[0],w[1],8)
   # print single_point_crossover(w[1],w[2],8)
   # print single_point_crossover(w[2],w[3],8)
   # print single_point_crossover(w[3],w[4],8)


   # print mutation(w[0],8)
   # print mutation(w[1],8)
   # print mutation(w[2],8)
   # print mutation(w[3],8)
   # print mutation(w[4],8)

   print GA(X,D)


def number_to_bits(n,bitcount):
   Bits = np.zeros(bitcount)
   if n < 0:
      Bits[0] = 1
      n = -n
   factor = 1.
   for k in range(1,bitcount):
      if factor <= n:
         Bits[k] = 1
         n = n - factor
      factor = factor / 2
   return Bits

def bits_to_number(Bits,bitcount):
   n = 0
   factor = 1.
   for k in range(1,bitcount):
      n = n + factor * Bits[k]
      factor = factor / 2
   if Bits[0] == 1:
      n = -n
   return n

def solution_to_bits(x,bitcount):
   Bits = []
   for i in range(0,len(x)):
      arr = number_to_bits(x[i],bitcount)
      for j in arr:
         Bits.append(j)
   return Bits

def bits_to_solution(Bits,numcount,bitcount):
   x = []
   for n in range(1,numcount+1):
      x.append(bits_to_number(Bits[(n-1)*bitcount:n*bitcount],bitcount))
   return x

def run_model(x,w):
   NET = w[0]
   for i in range(0,len(x)):
      NET = NET + x[i] * w[i+1]
   if NET < 0:
      y = 0
   elif NET > 1:
      y = 1
   else:
      y = NET
   return y

def error_function(X,W,D):
   E = []
   for i in range(0,len(W)):
      Y = []
      for k in range(0,len(X)):
         Y.append(run_model(X[k],W[i]))
      E.append(max(np.power(np.subtract(Y,D),2)))
   return E

def compute_probabilities(E):
   F = []
   for i in E:
      F.append(math.exp(-i))
   P = np.divide(F,sum(F))
   for k in range(1,len(P)-1):
      P[k] = P[k] + P[k-1]
   P[-1] = 1
   return P

def selection(Population,P,n):
   Selection = []
   while len(Selection) < n:
      r = np.random.random()
      num = 0
      while r > P[num]:
         num = num + 1
      Selection.append(Population[num])
   return Selection

def single_point_crossover(parent1, parent2,bitcount):
   p1 = solution_to_bits (parent1, bitcount)
   p2 = solution_to_bits (parent2, bitcount)
   r = np.random.randint(1,len(p1))
   c = p1[0:r+1] + p2[r+1:len(p1)]
   numcount = len(c)/bitcount
   child = bits_to_solution(c, numcount, bitcount)
   return child

def mutation(oldsolution,bitcount):
   s = solution_to_bits(oldsolution,bitcount)
   r = np.random.randint(1,len(s))
   s[r] = 1 - s[r]
   numcount = len(s)/bitcount
   newsolution = bits_to_solution(s,numcount,bitcount)
   return newsolution

def GA(X,D):
   bitcount = 8
   mutrate = 5
   e = 0.1
   maxit = 1000
   crossrate = 4
   ncount = 3
   popsize = 10
   W = []
   for i in range(0,popsize):
      W.append([])
      for j in range(0,ncount):
         W[i].append(np.random.random()-0.5)
   F = error_function(X,W,D)
   it = 0
   while it < maxit and min(F) > e:
      it = it + 1
      P = compute_probabilities(F)
      #(A)
      WNEXT = selection(W,P,popsize - crossrate)
      #(B)
      WPARENTS = selection(W,P,crossrate*2)
      while WPARENTS != []:
         p1 = WPARENTS.pop()
         p2 = WPARENTS.pop()
         child = single_point_crossover(p1,p2,bitcount)
         WNEXT.append(child)
      #(C1)
      WMUTATION = []
      memory = []
      for i in range(0,mutrate):
         while True:
            j = np.random.randint(0,len(W))
            if j not in memory:
               break;
         memory.append(j)
         WMUTATION.append(W[j])
      WMUTATION2 = []
      for ee in WMUTATION:
         WMUTATION2.append(mutation(ee,bitcount))
      #(C2)
      i = 0;
      for j in memory:
         WNEXT[j] = WMUTATION2[i]
         i = i + 1
      #(D)
      W = WNEXT
      F = error_function(X,W,D)
   Solution = W[np.argmin(F)]
   return Solution

if __name__ == "__main__":
   main()
