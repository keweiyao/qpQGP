import numpy as np
from scipy.special import kn
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt

Lambda2 = 0.2**2
Nc = 3.
nf = 3.
alpha0 = 4.*np.pi/(11. - 2./3.*nf)
Q2cut_l = -Lambda2 * np.exp(alpha0)
Q2cut_h = 0.0
pf_g = 4.*np.pi/3.*(Nc + nf/2.)*0.2
pf_q = np.pi/2.*(Nc*Nc - 1)/2./Nc

def alpha(T2):
    if T2 < Q2cut_l:
        return alpha0 / np.log( -T2/Lambda2 );
    elif T2 <= Q2cut_h:
        return 1.0
    else:
        return alpha0 * ( .5 - np.arctan( np.log(T2/Lambda2)/np.pi ) / np.pi );

alpha = np.vectorize(alpha)

class qpQGP():
	"""
	qpQGP object takes in Lattice-QCD calculation and make a quasiparticle interpretation
	Fit the temperature dependence of the thermal-mass of gluon, ud quark (isospin asymmetry) and s quark
	In linear Boltzmann model, one also makes a quasiparticle interpretation of the medium, this allows a consistent sampling of quasi-gluon, quasi-quarks that enters the elementaty scattering process.
	"""
	def __init__(self, eosfilepath):
		self.ptab, self.TrAntab, self.cs2tab, self.ttab = np.loadtxt(eosfilepath).T
		self.etab = 3.*self.ptab + self.TrAntab
		self.ttab *= 1e-3

	def normPuds(self, T, zq, mq):
		x = mq/T
		result = 0.
		zpow = 1.
		for n in range(1, 100):
			zpow *= zq
			if n%2 == 1:
				result += zpow*x**2*kn(2, n*x)/(n)**2
			else:
				result -= zpow*x**2*kn(2, n*x)/(n)**2
		result *= 18./np.pi**2/2.
		return result

	def normEuds(self, T, zq, mq):
		x = mq/T
		result = 0.
		zpow = 1.
		for n in range(1, 100):
			zpow *= zq
			if n%2 == 1:
				result += zpow*(kn(4, x*n) - kn(0, x*n))/8.*x**4
			else:
				result -= zpow*(kn(4, x*n) - kn(0, x*n))/8.*x**4
		result *= 18./np.pi**2/2.
		return result

	def normPg(self, T, zg, mg):
		x = mg/T
		result = 0.
		zpow = 1.
		for n in range(1, 100):
			zpow *= zg
			result += zpow*x**2*kn(2, n*x)/(n)**2
		result *= 16./np.pi**2/2.
		return result

	def normEg(self, T, zg, mg):
		x = mg/T
		result = 0.
		zpow = 1.
		for n in range(1, 100):
			zpow *= zg
			result += zpow*(kn(4, x*n) - kn(0, x*n))/8.*x**4
		result *= 16./np.pi**2/2.
		return result
	
	def normP(self, T, zq, zg, mq, mg):
		return self.normPg(T, zg, mg) + self.normPuds(T, zq, mq)+ self.normPuds(T, 1./zq, mq)
	
	def normE(self, T, zq, zg, mq, mg):
		return self.normEg(T, zg, mg) + self.normEuds(T, zq, mq)+ self.normEuds(T, 1./zq, mq)
	
fitter = qpQGP(sys.argv[1])
normP = np.vectorize(fitter.normP)
normE = np.vectorize(fitter.normE)
T = fitter.ttab


def diff(x, T, lP, lE):
	Mq = (pf_q * T**2 * alpha(T**2))**0.5
	Mg = (pf_g * T**2 * alpha(T**2))**0.5
	Zq, Zg = x
	nP = normP(T, Zq, Zg, Mq, Mg)
	nE = normE(T, Zq, Zg, Mq, Mg)	
	return (nE-lE)**2/lE**2 + (nP-lP)**2/lP**2

Zq = []
Zg = []
for i, t in enumerate(T):
	res = minimize(diff, [1., 1.], bounds=[[0, 2.], [0, 2.]], args = (t, fitter.ptab[i], fitter.etab[i]) )
	print res['x']
	Zq.append(res['x'][0])
	Zg.append(res['x'][1])

Mq = (pf_q * T**2 * alpha(T**2))**0.5
Mg = (pf_g * T**2 * alpha(T**2))**0.5
nP = normP(T, Zq, Zg, Mq, Mg)
nE = normE(T, Zq, Zg, Mq, Mg)	

plt.plot(T, fitter.ptab, 'ro')
plt.plot(T, nP, 'r-')

plt.plot(T, fitter.etab, 'bo')
plt.plot(T, nE, 'b-')

plt.plot(T, fitter.TrAntab, 'go')
plt.plot(T, nE-3.*nP, 'g-')
plt.show()




