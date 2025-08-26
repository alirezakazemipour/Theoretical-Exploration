
import matplotlib.pyplot as plt

class MazeRenderer:

    def __init__(self):
        self.initializedRender = False

    def initRender(self, states,actions,R,P,nameActions,current):
        ()



    def render(self,states,actions,R,P,nameActions,current,lastaction,lastreward):

        if (not self.initializedRender):
            self.initRender(states,actions,R,P,nameActions,current)
            self.initializedRender = True

        plt.figure(self.numFigure)
        row, col = self.from_s(self.s)
        v = self.maze[row][col]
        self.maze[row][col] = 1.5
        plt.imshow(self.maze, cmap='hot', interpolation='nearest')
        self.maze[row][col] = v
        plt.show(block=False)
        plt.pause(0.01)