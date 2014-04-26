from bottle import route, run, template
import urllib2
import numpy as np
from scipy.sparse import csc_matrix
from pprint import pprint

def pageRank(G, s = .85, maxerr = .001):
    """
    Computes the pagerank for each of the n states.

    Used in webpage ranking and text summarization using unweighted
    or weighted transitions respectively.


    Args
    ----------
    G: matrix representing state transitions
       Gij can be a boolean or non negative real number representing the
       transition weight from state i to j.

    Kwargs
    ----------
    s: probability of following a transition. 1-s probability of teleporting
       to another state. Defaults to 0.85

    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged. Defaults to 0.001
    """
    n = G.shape[0]

    # transform G into markov matrix M
    M = csc_matrix(G,dtype=np.float)
    rsums = np.array(M.sum(1))[:,0]
    ri, ci = M.nonzero()
    M.data /= rsums[ri]

    # bool array of sink states
    sink = rsums==0

    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in xrange(0,n):
            # inlinks of state i
            Ii = np.array(M[:,i].todense())[:,0]
            # account for sink states
            Si = sink / float(n)
            # account for teleportation to state i
            Ti = np.ones(n) / float(n)

            r[i] = ro.dot( Ii*s + Si*s + Ti*(1-s) )

    # return normalized pagerank
    return r/sum(r)




if __name__=='__main__':
    # Example extracted from 'Introduction to Information Retrieval'
    G = np.array([[0,0,1,0,0,0,0],
                  [0,1,1,0,0,0,0],
                  [1,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0],
                  [0,0,0,0,0,0,1],
                  [0,0,0,0,0,1,1],
                  [0,0,0,1,1,0,1]])

    print pageRank(G,s=.86)



@route('/')
def index():
  c = urllib2.urlopen('https://docs.google.com/spreadsheets/d/14h4ckElbR59fHxpNSdpebt_XlR4Wy4zux43E1QgyNMw/export?format=csv&id=14h4ckElbR59fHxpNSdpebt_XlR4Wy4zux43E1QgyNMw&gid=1242103736')
  data = c.read()
  arr = data.split('\n')
  titles = arr[0]
  indices = {}
  names = []
  teas = []

  for row in arr[1:]:
    date, winner, loser = row.split(',')
    teas.append((winner, loser))

  for tea in teas:
    winner, loser = tea
    if winner not in indices:
      indices[winner] = len(names)
      names.append(winner)
    if loser not in indices:
      indices[loser] = len(names)
      names.append(loser)

  wins = [ [ 0 for i in indices ] for i in indices ]
  losses = [ [ 0 for i in indices ] for i in indices]
  mat = [ [ 0 for i in indices ] for i in indices]

  for tea in teas:
    winner, loser = tea
    wins[indices[winner]][indices[loser]] += 1
    losses[indices[loser]][indices[winner]] += 1

  for winner in range(len(indices)):
    for loser in range(len(indices)):
      win_count = wins[winner][loser]
      loss_count = losses[winner][loser]
      # put an edge from the loser to the winner
      mat[loser][winner] = 0 if win_count == 0 else float(win_count) / (win_count + loss_count)
      print names[loser], 'to', names[winner], 0 if win_count == 0 else float(win_count) / (win_count + loss_count)

  print indices
  print names
  print wins
  print losses
  pprint(mat)
  G = np.array(mat)  
  scores = pageRank(G,s=.86)
  print scores

  labeled_scores = zip(names, scores)
  sorted_scores = sorted(labeled_scores, key=lambda s: s[1])

  response = ""
  for i in range(len(names)-1, -1, -1):
    response += str(len(names) - i) + ": " + str(sorted_scores[i][0]) + " (" + str(sorted_scores[i][1])+ ")<br/>"



  return response

run(host='localhost', port=8080)
