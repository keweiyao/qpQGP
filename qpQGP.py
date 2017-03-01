import numpy as np
from scipy.special import kn
from scipy.optimize import minimize, bisect
import sys
import matplotlib.pyplot as plt

Lambda2 = 0.2**2
Nc = 3.
nf = 3.
alpha0 = 4.*np.pi/(11. - 2./3.*nf)
Q2cut_l = -Lambda2 * np.exp(alpha0)
Q2cut_h = 0.0

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
		self.stab = self.ptab + self.etab
		self.ttab *= 1e-3
		self.Mq = np.zeros_like(self.ttab)
		self.Mg = np.zeros_like(self.ttab)
		self.B = np.zeros_like(self.ttab)

	def normPuds(self, T, mq):
		x = mq/T
		result = 0.
		for n in range(1, 100):
			if n%2 == 1:
				result += x**2*kn(2, n*x)/(n)**2
			else:
				result -= x**2*kn(2, n*x)/(n)**2
		result *= 36./np.pi**2/2.
		return result

	def normEuds(self, T, mq):
		x = mq/T
		result = 0.
		for n in range(1, 100):
			if n%2 == 1:
				result += (kn(4, x*n) - kn(0, x*n))/8.*x**4
			else:
				result -= (kn(4, x*n) - kn(0, x*n))/8.*x**4
		result *= 36./np.pi**2/2.
		return result

	def normNuds(self, T, mq):
		x = mq/T
		result = 0.
		for n in range(1, 100):
			if n%2 == 1:
				result += (n*x)**2*kn(2,n*x)/n**3
			else:
				result -=  (n*x)**2*kn(2,n*x)/n**3
		result *= 36./np.pi**2/2.
		return result

	def normNg(self, T, mg):
		x = mg/T
		result = 0.
		for n in range(1, 100):
			result += (n*x)**2*kn(2,n*x)/n**3
		result *= 16./np.pi**2/2.
		return result

	def normPg(self, T, mg):
		x = mg/T
		result = 0.
		for n in range(1, 100):
			result += x**2*kn(2, n*x)/(n)**2
		result *= 16./np.pi**2/2.
		return result

	def normEg(self, T, mg):
		x = mg/T
		result = 0.
		for n in range(1, 100):
			result += (kn(4, x*n) - kn(0, x*n))/8.*x**4
		result *= 16./np.pi**2/2.
		return result
	
	def normS(self, T, mq, mg):
		return self.normPg(T, mg) + self.normEg(T, mg) + self.normPuds(T, mq) + self.normEuds(T, mq)
	
	def normN(self, T, mq, mg):
		return self.normNg(T, mg) + self.normNuds(T, mq)

	def idealnormP(self, T, mq, mg):
		return self.normPg(T, mg) + self.normPuds(T, mq)

	def idealnormE(self, T, mq, mg):
		return self.normEg(T, mg) + self.normEuds(T, mq)
	
	def calculate_B(self):
		result = (self.etab[0] - self.idealnormE(self.ttab[0], self.Mq[0], self.Mg[0]))*self.ttab[0]**4
		self.B[0] = result/self.ttab[0]**4
		for i, T in enumerate(self.ttab):
			if i == len(self.ttab)-2:
				break
			mq, mg = self.Mq[i], self.Mg[i]
			result -= T**4*(self.normEuds(T, mq) - 3.*self.normPuds(T, mq))*(np.log(self.Mq[i+1])-np.log(self.Mq[i]) )
			result -= T**4*(self.normEg(T, mg) - 3.*self.normPg(T, mg))*(np.log(self.Mg[i+1])-np.log(self.Mg[i]) )
			self.B[i+1] = result/T**4
			
fitter = qpQGP(sys.argv[1])
normS = np.vectorize(fitter.normS)
idealnormE = np.vectorize(fitter.idealnormE)
idealnormP = np.vectorize(fitter.idealnormP)
normN = np.vectorize(fitter.normN)

T = fitter.ttab

ratio = np.sqrt(2.)
def diff(x, T, lS):
	nS = normS(T, x, x*ratio)
	return (nS-lS)**2/lS**2

Mq = []
xinit =0.3
for i, t in enumerate(T):
	res = minimize(diff, xinit, bounds=[[0,1.0]], args = (t, fitter.stab[i]) )
	print res['x'][0]
	xinit = res['x'][0]
	Mq.append(res['x'][0])

Mq = np.array(Mq)
Mg = Mq*ratio

fitter.Mq = Mq
fitter.Mg = Mg

fitter.calculate_B()

nS = normS(T, Mq, Mg)
nE = idealnormE(T, Mq, Mg)
nP = idealnormP(T, Mq, Mg)
nP = idealnormP(T, Mq, Mg)

N = normN(T, Mq, Mg)*T**3
N0 = normN(T, 1e-5, 1e-5)*T**3
plt.figure(figsize=(15,12))
plt.plot(T, fitter.stab, 'rs', label=r'HotQCD, $s/T^3$')
plt.plot(T, fitter.ptab, 'gs', label=r'HotQCD, $p/T^4$')
plt.plot(T, fitter.etab, 'bs', label=r'HotQCD, $e/T^4$')
plt.plot(T, nS, 'r-', linewidth=2., label=r'QP')
plt.plot(T, nP - fitter.B, 'g-', linewidth=2., label=r'QP, w/ $B(T)$')
plt.plot(T, nE + fitter.B, 'b-', linewidth=2., label=r'QP, w/ $B(T)$')

plt.plot(T, nP, 'g--', linewidth=2., label=r'QP, w/o $B(T)$')
plt.plot(T, nE, 'b--', linewidth=2, label=r'QP, w/o $B(T)$')

plt.legend(loc='best', framealpha=0., ncol=3, fontsize=20)
plt.xlabel(r'$T$ [GeV]', size=20)
#plt.ylabel(r'$M/T$', size=20)
plt.show()




def RHS(T, Mq, Mg):
	x1 = Mg/T
	x2 = Mq/T
	if x1 < 1e-9 and x2 < 1e-9:
		return (np.pi**2/3.*Nc + np.pi**2/6.*nf)*4./np.pi
	result1 = 0.
	result2 = 0.
	for n in range(1, 100):
		da1 = (2.*kn(1, x1*n)*x1*n + (x1*n)**2*kn(0, x1*n))/n**2
		da2 = (2.*kn(1, x2*n)*x2*n + (x2*n)**2*kn(0, x2*n))/n**2
		result1 += da1
		if n%2 == 1:
			result2 += da2
		else:
			result2 -= da2
	return (result1*Nc + result2*nf)*4./np.pi

def equation(mD, T, Mq, Mg):
	x = (mD/T)**2
	return x/alpha(x*T*T) - RHS(T, Mq, Mg)

mD = []
for i, t in enumerate(fitter.ttab):
	x0 = bisect(equation, 0., 10.,  args=(t, fitter.Mq[i], fitter.Mg[2]) )
	mD.append(x0)
	print x0
mD = np.array(mD)

mD0 = []
for i, t in enumerate(fitter.ttab):
	x0 = bisect(equation, 0., 10.,  args=(t, 0., 0.) )
	mD0.append(x0)
	print x0
mD0 = np.array(mD0)
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
#plt.plot(fitter.ttab, fitter.Mq/fitter.ttab, 'r-', linewidth=2., label=r'$M_q$')
#plt.plot(fitter.ttab, fitter.Mg/fitter.ttab, 'g-', linewidth=2., label=r'$M_g = \sqrt{3} M_q$')
plt.plot(fitter.ttab, fitter.Mq, 'r-', linewidth=2., label=r'$M_q$')
plt.plot(fitter.ttab, fitter.Mg, 'g-', linewidth=2., label=r'$M_g = \sqrt{3} M_q$')
plt.legend(loc='best', framealpha=0., ncol=1, fontsize=20)
plt.xlabel(r'$T$ [GeV]', size=20)
plt.ylabel(r'$M$ [GeV]', size=20)
plt.ylim(0., 1.5)
plt.subplot(1,2,2)
plt.plot(fitter.ttab/0.154, mD/fitter.ttab, 'b-', linewidth=2., label=r'$M_D$ with $M_q(T), M_g(T)$')
plt.plot(fitter.ttab/0.154, mD0/fitter.ttab, 'b--', linewidth=2., label=r'$M_D$ set $M_q = M_g = 0$')
#plt.plot(fitter.ttab, mD, 'b-', linewidth=2., label=r'$M_D$ with $M_q(T), M_g(T)$')
#plt.plot(fitter.ttab, mD0, 'b--', linewidth=2., label=r'$M_D$ set $M_q = M_g = 0$')
plt.legend(loc='best', framealpha=0., ncol=1, fontsize=20)
plt.xlabel(r'$T$ [GeV]', size=20)
plt.ylabel(r'$M_D$ [GeV]', size=20)
#plt.ylim(0., 1.5)
plt.show()

plt.plot(fitter.ttab, N0/mD0**2, 'r-', linewidth=2., label=r'$M_{q,g}=0$ + $m_D(M_{q,g}=0)$')
plt.plot(fitter.ttab, N/mD0**2, 'g-', linewidth=2., label=r'$M_{q,g(T)}$ + $m_D(M_{q,g}=0)$')
plt.plot(fitter.ttab, N/mD**2, 'b-', linewidth=2., label=r'$M_{q,g(T)}$ + $m_D(M_{q,g(T)})$')

plt.legend(loc='best', framealpha=0., ncol=1, fontsize=20)
plt.xlabel(r'$T$ [GeV]', size=20)
plt.ylabel(r'$n/m_D^2$ [GeV]', size=20)
plt.show()


