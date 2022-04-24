import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import norm
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc
rc('animation', html='jshtml')
from mpl_toolkits.mplot3d import Axes3D
## Sætter grænseværdien for animationsstørrelsen op##
matplotlib.rcParams['animation.embed_limit'] = 2**128

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#m1,m2,m3 = 3e9,3e9,3e9 ## masserne i enheder af kg
#G = 6.67430e-11   ## Den universelle grav. konst. i N*m^2*kg^-2

m1, m2, m3 = 1, 1, 1
G = 1

tinit = 0
tfinal = 100
trange = [tinit,tfinal]

##startbetingelser###
r1_0, r2_0, r3_0 = [-1,0,0], [1,0,1], [0,0,0]
v1_0, v2_0, v3_0 = [0.417701,0.303455,-0.41], [0.417701,0.303455,0.2], [-0.835402,-0.60691,0.4]


y_init = np.concatenate([r1_0,v1_0,r2_0,v2_0,r3_0,v3_0])

ts = np.linspace(tinit, tfinal, 100000)

def dydt(t,y):
    r1     = y[0:3]
    r1_dot = y[3:6]
    r2     = y[6:9]
    r2_dot = y[9:12]
    r3     = y[12:15]
    r3_dot = y[15:18]
    
    d_r1_dt = r1_dot
    d_r2_dt = r2_dot
    d_r3_dt = r3_dot
    
    d_r1_dot_dt = -G*m2*((r1-r2)/(norm(r1-r2)**3))-G*m3*((r1-r3)/(norm(r1-r3)**3))
    d_r2_dot_dt = -G*m3*((r2-r3)/(norm(r2-r3)**3))-G*m1*((r2-r1)/(norm(r2-r1)**3))
    d_r3_dot_dt = -G*m1*((r3-r1)/(norm(r3-r1)**3))-G*m2*((r3-r2)/(norm(r3-r2)**3))
    
    return np.concatenate([d_r1_dt,d_r1_dot_dt,d_r2_dt,d_r2_dot_dt,d_r3_dt,d_r3_dot_dt])

mysol = solve_ivp(dydt, trange, y_init, t_eval = ts,rtol=3e-14)

time_vals = mysol.t

r1_x, r1_y, r1_z = mysol.y[0], mysol.y[1], mysol.y[2]
v1_x, v1_y, v1_z = mysol.y[3], mysol.y[4], mysol.y[5]

r2_x, r2_y, r2_z = mysol.y[6], mysol.y[7], mysol.y[8]
v2_x, v2_y, v2_z = mysol.y[9], mysol.y[10], mysol.y[11]

r3_x, r3_y, r3_z = mysol.y[12], mysol.y[13], mysol.y[14]
v3_x, v3_y, v3_z = mysol.y[15], mysol.y[16], mysol.y[17]

plt.title('Bevægelseskurver for de 3 masser')
plt.plot(r1_x,r1_y), plt.plot(r2_x,r2_y), plt.plot(r3_x,r3_y)
plt.grid()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


fig, ax = plt.subplots(3,2,figsize = (15,15))
fig.suptitle('State-space plots af de 3 masser', fontsize=20)
ax[0][0].plot(v1_x,r1_x), ax[0][1].plot(v1_y,r1_y), ax[0][0].grid(),  ax[0][1].grid()
ax[1][0].plot(v2_x,r2_x), ax[1][1].plot(v2_y,r2_y), ax[1][0].grid(),  ax[1][1].grid()
ax[2][0].plot(v3_x,r3_x), ax[2][1].plot(v3_y,r3_y), ax[2][0].grid(),  ax[2][1].grid()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def shorten_frames(factor, r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, r3_x, r3_y, r3_z):
    x1, y1, z1 = [r1_x[0]], [r1_y[0]], [r1_z[0]]
    x2, y2, z2 = [r2_x[0]], [r2_y[0]], [r2_z[0]]
    x3, y3, z3 = [r3_x[0]], [r3_y[0]], [r3_z[0]]
    k = 0
    for i in range(len(r1_x)):
        k+=1
        if k == factor:
            x1.append(r1_x[i]), y1.append(r1_y[i]), z1.append(r1_z[i])
            x2.append(r2_x[i]), y2.append(r2_y[i]), z2.append(r2_z[i])
            x3.append(r3_x[i]), y3.append(r3_y[i]), z3.append(r3_z[i])
            k = 0
            
    return x1, y1, z1, x2, y2, z2, x3, y3, z3
    
r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, r3_x, r3_y, r3_z = shorten_frames(10, r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, r3_x, r3_y, r3_z)         

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(1,3,1, projection='3d')
ax2 = fig.add_subplot(1,3,2, projection='3d')
ax3 = fig.add_subplot(1,3,3, projection='3d')

## ax1 ##
mass1_line, = ax1.plot([],[],[],lw=2)
mass1_dot, = ax1.plot([],[],[],'o')

mass2_line, = ax1.plot([], [], [],lw=2)
mass2_dot, = ax1.plot([],[],[],'o')

mass3_line, = ax1.plot([], [], [],lw=2)
mass3_dot, = ax1.plot([],[],[],'o')

## ax2 ##
f2mass1_line, = ax2.plot([],[],[],lw=2)
f2mass1_dot, = ax2.plot([],[],[],'o')

f2mass2_line, = ax2.plot([], [], [],lw=2)
f2mass2_dot, = ax2.plot([],[],[],'o')

f2mass3_line, = ax2.plot([], [], [],lw=2)
f2mass3_dot, = ax2.plot([],[],[],'o')

## ax3 ##
f3mass1_line, = ax3.plot([],[],[],lw=2)
f3mass1_dot, = ax3.plot([],[],[],'o')

f3mass2_line, = ax3.plot([], [], [],lw=2)
f3mass2_dot, = ax3.plot([],[],[],'o')

f3mass3_line, = ax3.plot([], [], [],lw=2)
f3mass3_dot, = ax3.plot([],[],[],'o')




def init():
    mass1_line.set_data([], [])
    mass1_line.set_3d_properties([])
    mass1_dot.set_data([], [])
    mass1_dot.set_3d_properties([])
    
    mass2_line.set_data([], [])
    mass2_line.set_3d_properties([])
    mass2_dot.set_data([], [])
    mass2_dot.set_3d_properties([])
    
    mass3_line.set_data([], [])
    mass3_line.set_3d_properties([])
    mass3_dot.set_data([], [])
    mass3_dot.set_3d_properties([])
    
    f2mass1_line.set_data([], [])
    f2mass1_line.set_3d_properties([])
    f2mass1_dot.set_data([], [])
    f2mass1_dot.set_3d_properties([])
    
    f2mass2_line.set_data([], [])
    f2mass2_line.set_3d_properties([])
    f2mass2_dot.set_data([], [])
    f2mass2_dot.set_3d_properties([])
    
    f2mass3_line.set_data([], [])
    f2mass3_line.set_3d_properties([])
    f2mass3_dot.set_data([], [])
    f2mass3_dot.set_3d_properties([])
    
    f3mass1_line.set_data([], [])
    f3mass1_line.set_3d_properties([])
    f3mass1_dot.set_data([], [])
    f3mass1_dot.set_3d_properties([])
    
    f3mass2_line.set_data([], [])
    f3mass2_line.set_3d_properties([])
    f3mass2_dot.set_data([], [])
    f3mass2_dot.set_3d_properties([])
    
    f3mass3_line.set_data([], [])
    f3mass3_line.set_3d_properties([])
    f3mass3_dot.set_data([], [])
    f3mass3_dot.set_3d_properties([])
    
    return [mass1_line, mass2_line, mass3_line, mass1_dot, mass2_dot, mass3_dot,
           f2mass1_line, f2mass2_line, f2mass3_line, f2mass1_dot, f2mass2_dot, f2mass3_dot,
           f3mass1_line, f3mass2_line, f3mass3_line, f3mass1_dot, f3mass2_dot, f3mass3_dot,]

def update(i, mass1_line, mass2_line, mass3_line, mass1_dot, mass2_dot, mass3_dot,
              f2mass1_line, f2mass2_line, f2mass3_line, f2mass1_dot, f2mass2_dot, f2mass3_dot,
              f3mass1_line, f3mass2_line, f3mass3_line, f3mass1_dot, f3mass2_dot, f3mass3_dot,
              r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, r3_x, r3_y, r3_z):
    
    mass1_line.set_data(r1_x[:i+1], r1_y[:i+1])
    mass1_line.set_3d_properties(r1_z[:i+1])
    mass1_dot.set_data(r1_x[i], r1_y[i])
    mass1_dot.set_3d_properties(r1_z[i])
    
    mass2_line.set_data(r2_x[:i+1], r2_y[:i+1])
    mass2_line.set_3d_properties(r2_z[:i+1])
    mass2_dot.set_data(r2_x[i], r2_y[i])
    mass2_dot.set_3d_properties(r2_z[i])
    
    mass3_line.set_data(r3_x[:i+1], r3_y[:i+1])
    mass3_line.set_3d_properties(r3_z[:i+1])
    mass3_dot.set_data(r3_x[i], r3_y[i])
    mass3_dot.set_3d_properties(r3_z[i])
    
    f2mass1_line.set_data(r1_x[:i+1], r1_y[:i+1])
    f2mass1_line.set_3d_properties(r1_z[:i+1])
    f2mass1_dot.set_data(r1_x[i], r1_y[i])
    f2mass1_dot.set_3d_properties(r1_z[i])
    
    f2mass2_line.set_data(r2_x[:i+1], r2_y[:i+1])
    f2mass2_line.set_3d_properties(r2_z[:i+1])
    f2mass2_dot.set_data(r2_x[i], r2_y[i])
    f2mass2_dot.set_3d_properties(r2_z[i])
    
    f2mass3_line.set_data(r3_x[:i+1], r3_y[:i+1])
    f2mass3_line.set_3d_properties(r3_z[:i+1])
    f2mass3_dot.set_data(r3_x[i], r3_y[i])
    f2mass3_dot.set_3d_properties(r3_z[i])
    
    f3mass1_line.set_data(r1_x[:i+1], r1_y[:i+1])
    f3mass1_line.set_3d_properties(r1_z[:i+1])
    f3mass1_dot.set_data(r1_x[i], r1_y[i])
    f3mass1_dot.set_3d_properties(r1_z[i])
    
    f3mass2_line.set_data(r2_x[:i+1], r2_y[:i+1])
    f3mass2_line.set_3d_properties(r2_z[:i+1])
    f3mass2_dot.set_data(r2_x[i], r2_y[i])
    f3mass2_dot.set_3d_properties(r2_z[i])
    
    f3mass3_line.set_data(r3_x[:i+1], r3_y[:i+1])
    f3mass3_line.set_3d_properties(r3_z[:i+1])
    f3mass3_dot.set_data(r3_x[i], r3_y[i])
    f3mass3_dot.set_3d_properties(r3_z[i])
    

    return [mass1_line, mass2_line, mass3_line, mass1_dot, mass2_dot, mass3_dot,
           f2mass1_line, f2mass2_line, f2mass3_line, f2mass1_dot, f2mass2_dot, f2mass3_dot,
           f3mass1_line, f3mass2_line, f3mass3_line, f3mass1_dot, f3mass2_dot, f3mass3_dot,]

ax1.set_xlim(-1.8,1.8), ax1.set_ylim(-1.8,1.8), ax1.set_zlim(-0.6,4)
ax2.set_xlim(-1.8,1.8), ax2.set_ylim(-1.8,1.8), ax2.set_zlim(-0.6,4)
ax3.set_xlim(-1.8,1.8), ax3.set_ylim(-1.8,1.8), ax3.set_zlim(-0.6,4)

anim = animation.FuncAnimation(fig, 
                               update, 
                               init_func=init, 
                               fargs=(mass1_line, mass2_line, mass3_line,
                                      mass1_dot, mass2_dot, mass3_dot,
                                      f2mass1_line, f2mass2_line, f2mass3_line, f2mass1_dot, 
                                      f2mass2_dot, f2mass3_dot,f3mass1_line, f3mass2_line, 
                                      f3mass3_line, f3mass1_dot, f3mass2_dot, f3mass3_dot,
                                      r1_x, r1_y, r1_z, r2_x, r2_y, r2_z, r3_x, r3_y, r3_z),
                               frames=len(r1_x), 
                               interval=1,
                               repeat_delay=0, 
                               blit=True)

ax1.view_init(0,5)
ax2.view_init(0,45)
ax3.view_init(90,0)

    

