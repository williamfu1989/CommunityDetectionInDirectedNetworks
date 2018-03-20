
G = nx.read_graphml("/Users/Matter/paper_code/simrank/widom.graphml")
r=0.8
max_iter=10
eps=1e-4
nodes = G.nodes()   #['1', '0', '3', '2', '4']

pred_func = G.predecessors if isinstance(G, nx.DiGraph) else G.neighbors
#  nodes_i      {'0': 1, '1': 0, '2': 3, '3': 2, '4': 4}
nodes_i = {nodes[i]: i for i in range(0, len(nodes))}

sim_prev = numpy.zeros(len(nodes))  # array([ 0.,  0.,  0.,  0.,  0.])
sim = numpy.identity(len(nodes))
# ###  ????
# array([[ 1.,  0.,  0.,  0.,  0.],
#        [ 0.,  1.,  0.,  0.,  0.],
#        [ 0.,  0.,  1.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.,  0.],
#        [ 0.,  0.,  0.,  0.,  1.]])
# ###

# round 1
sim_prev = numpy.copy(sim)

for u, v in itertools.product(nodes, nodes):
    if u==v:continue
    u_ps, v_ps = pred_func(u), pred_func(v)
    s_uv = 0
    for u_n, v_n in itertools.product(u_ps, v_ps):
        s_uv += sim_prev[nodes_i[u_n]][nodes_i[v_n]]
    sim[nodes_i[u]][nodes_i[v]] = (r * s_uv) / (len(u_ps) * len(v_ps) + DIV_EPS)
