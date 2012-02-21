#!/usr/bin/env python2
import numpy as np

def main():
   tfile = open('T.txt', 'r')
   t = []
   for line in tfile:
     t.append(line[:-2].split("\t"))
   t = [[int(y) for y in x] for x in t]

   rfile = open('R.txt', 'r')
   r = []
   for line in rfile:
      r.append(line[:-2].split("\t"))
   r = [[int(y) for y in x] for x in r]


   pfile = open('P.txt', 'r')
   p = []
   for line in pfile:
      p.append(int(line[:-2]))

   ffile = open('F.txt', 'r')
   f = []
   for line in ffile:
      f.append(int(line[:-2]))
   
   print generate_episode_soft(t,r,p,0.1,f)
   print "========================================================"
   print monte_carlo_det(t,r,f,0.9,0.1,500,p)

   
def generate_episode_soft(T,R,pi,e,F):
   while True:
      s = np.random.randint(0,len(T)) 
      if F[s] is not 1:
         break
   Ep = [(0,0,s+1)]
   while F[s] is not 1: 
      a0 = pi[s]
      pr = []
      for i in range(1,5):
         if i != a0:
            pr.append(e)
         else:
            pr.append(1-e*(4-1))
      c = np.cumsum(pr) #cumulative probability vector
      x = np.random.random(1)[0] #random number in range 0..1
      a = np.searchsorted(c,x) #search number in vector 'n return index
      s2 = T[s][a]-1 
      r = R[s][a]
      Ep.append((a+1,r,s2+1))
      s = s2
   return Ep

def monte_carlo_det(T,R,F,y,e,maxit,p = None):
   #random politic generation
   if p == None:
      p = []
      for s in range(0,len(T)):
         p.append(np.random.randint(0,4))
   Returns = []
   Q = []
   #generation of empty array set for Returns
   #generation for zero array set for Q
   for s in range(0,len(T)):
      Returns.append([])
      Q.append([])
      for a in range(0,4):
         Returns[s].append([])
         Q[s].append(0)
   for i in range(0,maxit):
      Ep = generate_episode_soft(T,R,p,e,F)
      s = Ep[0][2] - 1
      memory = []
      for e2 in range(1,len(Ep)):
         a = Ep[e2][0] - 1
         s2 = Ep[e2][2] - 1
         if [a,s2] not in memory: #if first occurance
            memory.append([a,s2]) #add occurance to memory
            r = 0
            for f in range(e2,len(Ep))[::-1]: #backward iteration from ep end
               r = r * y + Ep[f][1]
            Returns[s][a].append(r)
            Q[s][a] = np.average(Returns[s][a])
         s = s2
      for si in range(0,len(T)):
         if F[si] is not 1:
            p[si] = np.argmax(Q[si]) + 1
   return p

if __name__ == "__main__":
   main()