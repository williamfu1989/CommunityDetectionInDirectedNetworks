import networkx as nx
import cmty
import rank

def main(argv):
  if len(argv) < 2:
    sys.stderr.write("Usage: %s <input graph>\n" % (argv[0],))
    return 1
  graph_fn = argv[1]
  G = nx.Graph()  #let's create the graph first
  buildG(G, graph_fn, ',')
  
  if _DEBUG_:
    print 'G nodes:', G.nodes()
    print 'G no of nodes:', G.number_of_nodes()
  
  n = G.number_of_nodes()    #|V|
  A = nx.adj_matrix(G)    #adjacenct matrix

  m_ = 0.0    #the weighted version for number of edges
  for i in range(0,n):
    for j in range(0,n):
      m_ += A[i,j]
  m_ = m_/2.0
  if _DEBUG_:
    print "m: %f" % m_

  #calculate the weighted degree for each node
  Orig_deg = {}
  Orig_deg = UpdateDeg(A, G.nodes())

  #run Newman alg
  runGirvanNewman(G, Orig_deg, m_)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
