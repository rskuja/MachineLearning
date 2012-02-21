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

   print evaluate_policy_det(t,r,p,0.9,0.1)
   print iterate_policy_det(t,r,p,0.9,0.1)

def evaluate_policy_det(T,R,p,y,o):
   v = np.zeros(len(T))
   while True:
      w = np.copy(v)
      d = 0
      for s in range(0,len(T)):
         a = p[s] - 1
         s2 = T[s][a] - 1
         r = R[s][a]
         v[s] = r + y * w[s2]
         d = max(d,abs(v[s]-w[s]))
      if not d >= o:
         break
   return v

def iterate_policy_det(T,R,p,y,o):
   while True:
      V = evaluate_policy_det(T,R,p,y,o)
      policy_stable = True
      for s in range(0,len(T)):
         oldp = p[s]
         maxval = None
         for a in range(0,4):
            s2 = T[s][a] - 1
            r = R[s][a]
            v = r + y * V[s2]
            if maxval is None or v > maxval:
               maxval = v
               p[s] = a + 1
         if p[s] != oldp:
            policy_stable = False
      if policy_stable:
         break
   return p 

if __name__ == "__main__":
   main()