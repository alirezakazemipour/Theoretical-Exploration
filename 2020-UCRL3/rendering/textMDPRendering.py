
import sys
from six import StringIO
from gym import utils
import string

class textRenderer:

    def __init__(self):
        self.initializedRender = False


    def initRender(self, states, actions, R, P, nameActions, current):
        self.nameActions = nameActions
        if(len(nameActions)!=len(actions)):
            self.nameActions = list(string.ascii_uppercase)[0:min(len(actions),26)]

    def render(self,states,actions,R,P,nameActions,current,lastaction,lastreward):

        if (not self.initializedRender):
            self.initRender(states,actions,R,P,nameActions,current)
            self.initializedRender = True

        # Print the MDp in text mode.
        # Red  = current state
        # Blue = all states accessible from current state (by playing some action)
        outfile = sys.stdout
        #outfile = StringIO() if mode == 'ansi' else sys.stdout

        desc = [str(s) for s in states]

        desc[current] = utils.colorize(desc[current], "red", highlight=True)
        for a in actions:
            for ssl in P[current][a]:
                if (ssl[0] > 0):
                    desc[ssl[1]] = utils.colorize(desc[ssl[1]], "blue", highlight=True)

        desc.append(" \t\tr=" + str(lastreward))

        if lastaction is not None:
            outfile.write("  ({})\t".format(self.nameActions[lastaction % 26]))
        else:
            outfile.write("\n")
        outfile.write("".join(''.join(line) for line in desc) + "\n")

        #if mode != 'text':
        #    return outfile