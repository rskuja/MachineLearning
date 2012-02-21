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
   
   print "Sarsa"
   print sarsa(t,r,f,0.9,0.1,200,0.1,p)
   print "================================"
   print "Q learning"
   print q_learning(t,r,f,0.9,0.1,200,0.1,p)
   
def next_state(T,R,s,e,p):
   a0 = p[s]
   pr = []
   for i in range(1,5):
      if i != a0:
         pr.append(e)
      else:
         pr.append(1-e*(4-1))
   c = np.cumsum(pr) #cumulative probability vector
   x = np.random.random(1)[0] #random number in range 0..1
   a = np.searchsorted(c,x) #search number in vector/ return index
   s2 = T[s][a]-1
   r = R[s][a]
   return a,r,s2

def sarsa(T,R,F,y,e,maxit,alf,p = None):
   #random politic generation
   if p is None:
      p = []
      for s in range(0,len(T)):
         p.append(np.random.randint(0,4))
   Q = []
   #generation for zero array set for Q
   for s in range(0,len(T)):
      Q.append([])
      for a in range(0,4):
         Q[s].append(0)
   for i in range(0,maxit):
      while True:
         s = np.random.randint(0,len(T)) 
         if F[s] is not 1:
            break
      a, r, s2 = next_state(T,R,s,e,p)
      while F[s] is not 1:
         if F[s2] is 1:
            Q[s][a] = Q[s][a] + alf * (r - Q[s][a])
            s = s2
         else:
            a2, r2, s3 = next_state(T,R,s2,e,p)
            Q[s][a] = Q[s][a] + alf * (r + y * Q[s2][a2] - Q[s][a])
            a, r, s, s2 = (a2, r2, s2, s3)
      for si in range(0,len(T)):
         if F[si] is not 1:
            p[si] = np.argmax(Q[si]) + 1
   return p

def q_learning(T,R,F,y,e,maxit,alf,p = None):
   #random politic generation
   if p is None:
      p = []
      for s in range(0,len(T)):
         p.append(np.random.randint(0,4))
   Q = []
   #generation for zero array set for Q
   for s in range(0,len(T)):
      Q.append([])
      for a in range(0,4):
         Q[s].append(0)
   for i in range(0,maxit):
      while True:
         s = np.random.randint(0,len(T)) 
         if F[s] is not 1:
            break
      while F[s] is not 1:
         a, r, s2 = next_state(T,R,s,e,p)
         if F[s2] is 1:
            Q[s][a] = Q[s][a] + alf * (r - Q[s][a])
         else:
            Q[s][a] = Q[s][a] + alf * (r + y * max(Q[s2]) - Q[s][a])
         s = s2
      for si in range(0,len(T)):
         if F[si] is not 1:
            p[si] = np.argmax(Q[si]) + 1
   return p

if __name__ == "__main__":
   main()