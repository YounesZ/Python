import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10 * np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

for phase in np.linspace(0, 10 * np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()


[[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]

self.racers[0].action_chain[0] = 0
rew_pos  =  [self.compute_displacement(self.racers[x].position_chain[-2], self.racers[x].velocities[self.racers[x].velocity_chain[-2]]) if y else [] for x,y in zip(range(len(self.racers)), race_on)]
reward  =   rew_pos[0][0]
self.racers[0].position_chain[-1] = list(rew_pos[0][1])
Qold    =   self.racers[0].action_value[self.racers[0].position_chain[-2][0], self.racers[0].position_chain[-2][1], self.racers[0].velocity_chain[-2], self.racers[0].action_chain[-2]]
Qnew    =   self.racers[0].action_value[self.racers[0].position_chain[-1][0], self.racers[0].position_chain[-1][1], self.racers[0].velocity_chain[-1], self.racers[0].action_chain[-1]]
print(Qold + self.racers[0].learnRate * (reward + self.racers[0].discount * Qnew - Qold))



for i in range(21):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.flush()
    sleep(0.25)


import progressbar
from time import sleep
bar = progressbar.ProgressBar(maxval=20, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ', 'just testing'])
bar.start()
for i in range(20):
    bar.update(i+1)
    sleep(0.1)
bar.finish()





import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=50, blit=True)
line_ani.save('lines.mp4', writer=writer)

fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                   blit=True)
im_ani.save('im.mp4', writer=writer)






self.ax2    =   self.figId.add_subplot(132);    self.ax2.title.set_text('Policy')
            self.ax3    =   self.figId.add_subplot(133);    self.ax3.title.set_text('Values')

self.imagePanels.append( self.ax2.imshow( np.dstack( [m1]*3 ), interpolation='nearest') )
            self.imagePanels.append( self.ax3.imshow( np.dstack( [m1]*3 ), interpolation='nearest') )

self.view_policy(-1)
self.view_value(-1)


def view_policy(self, cnt):
    # --- Draw the action arrows
    # Slice the policy
    for iy in range(self.maze_dims[0]):
        for ix in range(self.maze_dims[1]):
            pSlice = self.agents[0].policy[iy, ix, :]
            # Compute resultant vectors along each dimension
            resV = np.argmax(pSlice)
            ampV = 0
            if sum(pSlice) > 0:
                ampV = pSlice[resV] / sum(pSlice)
            # Draw arrows
            try:
                self.arrowsP[iy][ix][0].remove()
            except:
                pass
            iAct = self.agents[0].actions[resV]
            self.arrowsP[iy][ix] = [
                self.ax2.arrow(-iAct[1] / 2 + ix, -iAct[0] / 2 + iy, iAct[1] / 2, iAct[0] / 2, head_width=0.5 * ampV,
                               head_length=max(max(abs(np.array(iAct))) / 2, 0.001) * ampV, fc='k', ec='k')]


def view_value(self, cnt):
    # --- Draw the action arrows
    for iy in range(self.maze_dims[0]):
        for ix in range(self.maze_dims[1]):
            pSlice = self.agents[0].global_value[iy, ix, :]
            # Compute resultant vectors along each dimension
            indV = [np.multiply(x, y) for x, y in zip(pSlice, self.agents[0].actions)]
            resV = np.sum(np.array(indV), axis=0)
            scl = np.sum(abs(np.array(indV)), axis=0)
            scl = [1 if x == 0 else x for x in scl]
            resV = np.divide(resV, scl)
            ampV = np.sqrt(np.sum(resV ** 2))
            # Draw arrows
            try:
                self.arrowsV[iy][ix][0].remove()
            except:
                pass
            self.arrowsV[iy][ix] = [self.ax3.arrow(-resV[1] / 2 + ix, -resV[0] / 2 + iy, resV[1] / 2, resV[0] / 2,
                                                   head_width=0.5 * ampV, head_length=max(ampV / 2, 0.1), fc='k',
                                                   ec='k')]
