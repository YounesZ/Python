""" This function returns the unique elements of a redundant sequence while preserving the order in which the appear
in the original list"""


def main(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   index  = []
   count  = -1
   for item in seq:
       count+=1
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
       index.append(count)
   return result, index