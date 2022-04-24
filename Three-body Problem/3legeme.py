import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import norm
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
## Sætter grænseværdien for animationsstørrelsen op##
matplotlib.rcParams['animation.embed_limit'] = 2**128

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#m1,m2,m3 = 3e9,3e9,3e9 ## masserne i enheder af kg
#G = 6.67430e-11   ## Den universelle grav. konst. i N*m^2*kg^-2

m1, m2, m3 = 1, 1, 1
G = 1

tinit = 0
tfinal = 40
trange = [tinit,tfinal]

##startbetingelser###
r1_0, r2_0, r3_0 = [-1,0], [1,0], [0,0]
v1_0, v2_0, v3_0 = [0.417701,0.303455], [0.417701,0.303455], [-0.835402,-0.60691]

#r1_0, r2_0, r3_0 = [-0.97000436,0.24308753], [0,0], [0.97000436,-0.24308753]
#v1_0, v2_0, v3_0 = [0.4662036850,0.4323657300], [-0.93240737,-0.86473146], [0.4662036850,0.4323657300]

y_init = np.concatenate([r1_0,v1_0,r2_0,v2_0,r3_0,v3_0])

ts = np.linspace(tinit, tfinal, 10000)

def dydt(t,y):
    r1     = y[0:2]
    r1_dot = y[2:4]
    r2     = y[4:6]
    r2_dot = y[6:8]
    r3     = y[8:10]
    r3_dot = y[10:12]
    
    d_r1_dt = r1_dot
    d_r2_dt = r2_dot
    d_r3_dt = r3_dot
    
    d_r1_dot_dt = -G*m2*((r1-r2)/(norm(r1-r2)**3))-G*m3*((r1-r3)/(norm(r1-r3)**3))
    d_r2_dot_dt = -G*m3*((r2-r3)/(norm(r2-r3)**3))-G*m1*((r2-r1)/(norm(r2-r1)**3))
    d_r3_dot_dt = -G*m1*((r3-r1)/(norm(r3-r1)**3))-G*m2*((r3-r2)/(norm(r3-r2)**3))
    
    return np.concatenate([d_r1_dt,d_r1_dot_dt,d_r2_dt,d_r2_dot_dt,d_r3_dt,d_r3_dot_dt])

mysol = solve_ivp(dydt, trange, y_init, t_eval = ts,rtol=3e-14)

time_vals = mysol.t

r1_x, r1_y = mysol.y[0], mysol.y[1]
v1_x, v1_y = mysol.y[2], mysol.y[3]

r2_x, r2_y = mysol.y[4], mysol.y[5]
v2_x, v2_y = mysol.y[6], mysol.y[7]

r3_x, r3_y = mysol.y[8], mysol.y[9]
v3_x, v3_y = mysol.y[10], mysol.y[11]

plt.title('Bevægelseskurver for de 3 masser')
plt.plot(r1_x,r1_y), plt.plot(r2_x,r2_y), plt.plot(r3_x,r3_y)
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


fig, ax = plt.subplots(3,2,figsize = (15,15))
fig.suptitle('State-space plots af de 3 masser', fontsize=20)
ax[0][0].plot(v1_x,r1_x), ax[0][1].plot(v1_y,r1_y), ax[0][0].grid(),  ax[0][1].grid()
ax[1][0].plot(v2_x,r2_x), ax[1][1].plot(v2_y,r2_y), ax[1][0].grid(),  ax[1][1].grid()
ax[2][0].plot(v3_x,r3_x), ax[2][1].plot(v3_y,r3_y), ax[2][0].grid(),  ax[2][1].grid()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


fig, ax = plt.subplots(figsize=(10, 10))

mass1_line, = ax.plot([], [])
mass1_dot, = ax.plot([],[],'ro')

mass2_line, = ax.plot([], [])
mass2_dot, = ax.plot([],[],'ro')

mass3_line, = ax.plot([], [])
mass3_dot, = ax.plot([],[],'ro')


def update(i):
    mass1_line.set_data(r1_x[0:i+1], r1_y[0:i+1])
    mass1_dot.set_data(r1_x[i],r1_y[i])
    
    mass2_line.set_data(r2_x[0:i+1], r2_y[0:i+1])
    mass2_dot.set_data(r2_x[i],r2_y[i])
    
    mass3_line.set_data(r3_x[0:i+1], r3_y[0:i+1])
    mass3_dot.set_data(r3_x[i],r3_y[i])
    return mass1_line, mass1_dot, mass2_line, mass2_dot, mass3_line, mass3_dot


ax.set_xlim([-1.6, 1.6]), ax.set_ylim([-0.9, 0.9]), ax.set_aspect('equal')
ax.hlines(0,-1.6,1.6), ax.vlines(0,-0.9,0.9), ax.grid()

anim = animation.FuncAnimation(fig,
                               update,
                               frames=len(time_vals),
                               interval=1,
                               blit=True,
                               repeat_delay=0)
plt.show()