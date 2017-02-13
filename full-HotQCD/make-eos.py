import numpy as np
import matplotlib.pyplot as plt

e = np.loadtxt("energy.dat")
p = np.loadtxt("pressure.dat")
s = np.loadtxt("entropy.dat")
t = np.loadtxt("temperature.dat")
cv = np.loadtxt("cv.dat")
cs2 = np.loadtxt("cs2.dat")
theta = np.loadtxt("theta.dat")

f = open('HotQCD.dat', 'w')
for i in range(len(e)):
	f.write("%f %f %f %f\n"%(p[i], theta[i], cs2[i], t[i]))
f.close()

plt.plot(t, e, label='e')
plt.plot(t, p, label='p')
plt.plot(t, s, label='s')
plt.plot(t, cv, label='cv')
plt.plot(t, cs2, label='cs2')
plt.plot(t, theta, label='theta')
plt.legend()
plt.show()

