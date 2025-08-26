
import pydot
#import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class pydotRenderer:
    def __init__(self):
        self.initializedRender = False

    def initRender(self,states,actions,R,P,nameActions,current):

        G = pydot.Dot(graph_type='digraph')
        # colors = ['green', 'red', 'blue', 'orange', 'purple']
        colors = ['#FF0000', '#00FF00', '#0000FF', '#888800', '#008888', '#880088', '#555555']

        nodes = []
        for s in states:
            fillcolor = "#AAAAAA"
            if (s == current):
                fillcolor = "#FFAA00"
            nodes.append(pydot.Node(str(s), style="filled", fillcolor=fillcolor))
            G.add_node(nodes[-1])
        for s in states:
            for a in actions:
                for ssl in P[s][a]:  # ssl = (p(s),s, 'done')
                    label = ""
                    if (R[s][a].mean() > 0):
                        label = str("{:01.2f}".format(R[s][a].mean()))
                    col = colors[a % len(colors)]
                    r1 = int(col[1:3], 16)
                    g1 = int(col[3:5], 16)
                    b1 = int(col[5:7], 16)
                    # print("a:",a,r1,g1,b1,ssl[0])
                    r2 = (hex((int)(r1 * (ssl[0]))))[2:]
                    g2 = (hex((int)(g1 * (ssl[0]))))[2:]
                    b2 = (hex((int)(b1 * (ssl[0]))))[2:]
                    if (len(r2) == 1):
                        r2 = '0' + r2[0]
                    if (len(g2) == 1):
                        g2 = '0' + g2[0]
                    if (len(b2) == 1):
                        b2 = '0' + b2[0]
                    col = col[0] + r2 + g2 + b2
                    # print("a:",a,col)
                    e = pydot.Edge(nodes[s], nodes[ssl[1]], label=label, color=col, weight=ssl[0])
                    G.add_edge(e)
        self.G = G
        # G_str=G.create_png(prog='dot')
        # G_im = IPython.display.Image(G_str)
        # IPython.display(G_im)
        # display(pl)
        # print(pydot.EDGE_ATTRIBUTES)
        G.write_png('MDP-pydot-rendering.png')

    def render(self, states,actions,R,P,nameActions,current,lastaction,lastreward):
        if (not self.initializedRender):
            self.initRender(states,actions,R,P,nameActions,current)
            self.initializedRender = True
        for s in states:
            fillcolor = "#AAAAAA"
            if (s == current):
                fillcolor = "#FFAA00"
            n = (self.G.get_node(str(s)))[0]
            # print(n, "(before) ")
            n.set("fillcolor", fillcolor)
            # print(n,"(after)")
        self.G.write_png('MDP-pydot-rendering.png')
        # plt.pause(3)