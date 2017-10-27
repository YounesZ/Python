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






