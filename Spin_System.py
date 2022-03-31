import numpy as np
import multiprocessing as mp

# Spin operators and their derivatives

def J_Z(x, y, S):
    return S*(1-x**2-y**2)/(1+x**2+y**2)

def J_X(x, y, S):
    return 2*S*x/(1+x**2+y**2)

def J_Y(x, y, S):
    return 2*S*y/(1+x**2+y**2)

# XorY gives which variable the derivative is being taken with respect to
def J_Z_derivative(XorY, x, y, S):
    if XorY == 0:
        return -4*S*x/(1+x**2+y**2)**2
    else:
        return -4*S*y/(1+x**2+y**2)**2

def J_X_derivative(XorY, x, y, S):
    if XorY == 0:
        return 2*S*(y**2-x**2+1)/(1+x**2+y**2)**2
    else:
        return -4*S*x*y/(1+x**2+y**2)**2


def J_Y_derivative(XorY, x, y, S):
    if XorY == 0:
        return -4*S*x*y/(1+x**2+y**2)**2
    else:
        return 2*S*(x**2-y**2+1)/(1+x**2+y**2)**2

def J_Z_second_derivative(XorY_1, XorY_2, x, y, S):
    if XorY_1 == 0:
        if XorY_2 == 0:
            return 4*S*(3*x**2-y**2-1)/(x**2+y**2+1)**3
        else:
            return 16*S*x*y/(x**2+y**2+1)**3
    else:
        if XorY_2 == 0:
            return 16*S*x*y/(x**2+y**2+1)**3
        else:
            return 4*S*(3*y**2-x**2-1)/(x**2+y**2+1)**3     


def J_X_second_derivative(XorY_1, XorY_2, x, y, S):
    if XorY_1 == 0:
        if XorY_2 == 0:
            return 4*S*x*(x**2-3*y**2-3)/(x**2+y**2+1)**3
        else:
            return -4*S*y*(y**2-3*x**2+1)/(x**2+y**2+1)**3   
    else:
        if XorY_2 == 0:
            return -4*S*y*(y**2-3*x**2+1)/(x**2+y**2+1)**3
        else:
            return -4*S*x*(x**2-3*y**2+1)/(x**2+y**2+1)**3     

def J_Y_second_derivative(XorY_1, XorY_2, x, y, S):
    if XorY_1 == 0:
        if XorY_2 == 0:
            return -4*S*y*(y**2-3*x**2+1)/(x**2+y**2+1)**3
        else:
            return -4*S*x*(x**2-3*y**2+1)/(x**2+y**2+1)**3     
    else:
        if XorY_2 == 0:
            return -4*S*x*(x**2-3*y**2+1)/(x**2+y**2+1)**3
        else:
            return 4*S*y*(y**2-3*x**2-3)/(x**2+y**2+1)**3

#General abstract class for hamiltonians

class Hamiltonian:
    def __init__(self, num_particles, spin):
        self.S = spin
        self.N = num_particles

    def energy(self, X, Y):
        pass
    def derivative(self, XorY, particle_index, X, Y):
        pass
    def second_derivative(self, XorY_1, XorY_2, particle_index_1, particle_index_2, X, Y):
        pass

# Class for single spin hamiltonians with J_Y as the operator
class Single_Spin_Hamiltonian(Hamiltonian):
    def __init__(self, num_particles, spin):
        super().__init__(num_particles, spin)

    def energy(self, X, Y):
        return J_Y(X[0], Y[0], self.S[0])
    
    # particle_index indicates with respect to which particle you're taking the x or y partial
    def derivative(self, XorY, particle_index, X, Y):
        return J_Y_derivative(XorY, X[0], Y[0], self.S[0])
                
    def second_derivative(self, XorY_1, XorY_2, particle_index_1, particle_index_2, X, Y):
        return J_Y_second_derivative(XorY_1, XorY_2, X[0], Y[0], self.S[0])


# Class for frustrated spin triplet
class Frustrated_Triplet_Hamiltonian(Hamiltonian):
    def __init__(self, num_particles, spin, J, Gamma):
        self.G = Gamma
        self.J = J
        super().__init__(num_particles, spin)

    def energy(self, X, Y):
        E = 0
        E += self.G*(J_X(X[0],Y[0],self.S[0])*J_X(X[1],Y[1],self.S[1]) + 
                     J_X(X[0],Y[0],self.S[0])*J_X(X[2],Y[2],self.S[2]) +
                     J_X(X[1],Y[1],self.S[1])*J_X(X[2],Y[2],self.S[2]))
        E += self.J*(J_Z(X[0],Y[0],self.S[0])*J_Z(X[1],Y[1],self.S[1]) + 
                     J_Z(X[0],Y[0],self.S[0])*J_Z(X[2],Y[2],self.S[2]) +
                     J_Z(X[1],Y[1],self.S[1])*J_Z(X[2],Y[2],self.S[2]))
        return E
    
    #particle_index works the same as above
    
    def derivative(self, XorY, particle_index, X, Y):
        if particle_index == 2:
            particle_index = -1
        D = 0
        D += self.G*(J_X_derivative(XorY, X[particle_index], Y[particle_index], self.S[particle_index])
                     * J_X(X[particle_index+1], Y[particle_index+1], self.S[particle_index+1])
                     + J_X_derivative(XorY, X[particle_index], Y[particle_index], self.S[particle_index])
                     * J_X(X[particle_index-1], Y[particle_index-1], self.S[particle_index-1]))
        D += self.J*(J_Z_derivative(XorY, X[particle_index], Y[particle_index], self.S[particle_index])
                     * J_Z(X[particle_index+1], Y[particle_index+1], self.S[particle_index+1])
                     + J_Z_derivative(XorY, X[particle_index], Y[particle_index], self.S[particle_index])
                     * J_Z(X[particle_index-1], Y[particle_index-1], self.S[particle_index-1]))
        return D

    def second_derivative(self, XorY_1, XorY_2, particle_index_1, particle_index_2, X, Y):
        D = 0
        if particle_index_1 == 2:
            particle_index_1 = -1
        if particle_index_2 == 2:
            particle_index_2 = -1

        if particle_index_1 == particle_index_2:
            D = 0
            D += self.G*(J_X_second_derivative(XorY_1, XorY_2, X[particle_index_1], Y[particle_index_1], self.S[particle_index_1])
                         * J_X(X[particle_index_1+1], Y[particle_index_1+1], self.S[particle_index_1+1])
                         + J_X_second_derivative(XorY_1, XorY_2, X[particle_index_1], Y[particle_index_1], self.S[particle_index_1])
                         * J_X(X[particle_index_1-1], Y[particle_index_1-1], self.S[particle_index_1-1]))
            D += self.J*(J_Z_second_derivative(XorY_1, XorY_2, X[particle_index_1], Y[particle_index_1], self.S[particle_index_1])
                         * J_Z(X[particle_index_1+1], Y[particle_index_1+1], self.S[particle_index_1+1])
                         + J_Z_second_derivative(XorY_1, XorY_2, X[particle_index_1], Y[particle_index_1], self.S[particle_index_1])
                         * J_Z(X[particle_index_1-1], Y[particle_index_1-1], self.S[particle_index_1-1]))
            return D
        else:
            D = 0
            D += self.G*(J_X_derivative(XorY_1, X[particle_index_1], Y[particle_index_1], self.S[particle_index_1])
                         * J_X_derivative(XorY_2, X[particle_index_2], Y[particle_index_2], self.S[particle_index_2]))
            D += self.J*(J_Z_derivative(XorY_1, X[particle_index_1], Y[particle_index_1], self.S[particle_index_1])
                         * J_Z_derivative(XorY_2, X[particle_index_2], Y[particle_index_2], self.S[particle_index_2]))
            return D
class Four_site_lattice(Hamiltonian):
    def __init__(self, spin, J, Gamma):
	    self.G = Gamma
	    self.J = J
	    super().__init__(4, spin)
    def energy(self, X, Y):
            E = 0
            E += self.G*(J_X(X[0],Y[0],self.S[0])*J_X(X[1],Y[1],self.S[1]) +
                         J_X(X[0],Y[0],self.S[0])*J_X(X[2],Y[2],self.S[2]) +
                         J_X(X[1],Y[1],self.S[1])*J_X(X[2],Y[2],self.S[2]) + 
                         J_X(X[1],Y[1],self.S[1])*J_X(X[3],Y[3],self.S[3]) +
                         J_X(X[2],Y[2],self.S[2])*J_X(X[3],Y[3],self.S[3]))
            E += self.J*(J_Z(X[0],Y[0],self.S[0])*J_Z(X[1],Y[1],self.S[1]) +
                         J_Z(X[0],Y[0],self.S[0])*J_Z(X[2],Y[2],self.S[2]) +
                         J_Z(X[1],Y[1],self.S[1])*J_Z(X[2],Y[2],self.S[2]) + 
                         J_Z(X[1],Y[1],self.S[1])*J_Z(X[3],Y[3],self.S[3]) +
                         J_Z(X[2],Y[2],self.S[2])*J_Z(X[3],Y[3],self.S[3]))
            return E
    def derivative(self, XorY, particle_index, X, Y):
            D = 0
            if particle_index == 0:
                    D += ((self.G * J_X_derivative(XorY, X[0], Y[0], self.S[0])) * 
                         (J_X(X[1], Y[1], self.S[1]) + J_X(X[2], Y[2], self.S[2])))
                    D += (self.J * J_Z_derivative(XorY, X[0], Y[0], self.S[0]) * 
                         (J_Z(X[1], Y[1], self.S[1]) + J_Z(X[2], Y[2], self.S[2])))
            elif particle_index == 1:
                    D += (self.G * J_X_derivative(XorY, X[1], Y[1], self.S[1]) * 
                         (J_X(X[0], Y[0], self.S[0]) + J_X(X[2], Y[2], self.S[2] + J_X(X[3], Y[3], self.S[3]))))
                    D += (self.J * J_Z_derivative(XorY, X[1], Y[1], self.S[1]) * 
                         (J_Z(X[0], Y[0], self.S[0]) + J_Z(X[2], Y[2], self.S[2])  + J_Z(X[3], Y[3], self.S[3])))
            elif particle_index == 2:
                    D += (self.G * J_X_derivative(XorY, X[2], Y[2], self.S[2]) * 
                         (J_X(X[0], Y[0], self.S[0]) + J_X(X[1], Y[1], self.S[1] + J_X(X[3], Y[3], self.S[3]))))
                    D += (self.J * J_Z_derivative(XorY, X[2], Y[2], self.S[2]) * 
                         (J_Z(X[0], Y[0], self.S[0]) + J_Z(X[1], Y[1], self.S[1])  + J_Z(X[3], Y[3], self.S[3])))
            elif particle_index == 3:
                    D += (self.G * J_X_derivative(XorY, X[3], Y[3], self.S[3]) * 
                         (J_X(X[1], Y[1], self.S[1]) + J_X(X[2], Y[2], self.S[2])))
                    D += (self.J * J_Z_derivative(XorY, X[3], Y[3], self.S[3]) * 
                         (J_Z(X[1], Y[1], self.S[1]) + J_Z(X[2], Y[2], self.S[2])))
            return D

    def second_derivative(self, XorY_1, XorY_2, particle_index_1, particle_index_2, X, Y):
            D = 0
            if particle_index_1 == particle_index_2:
                    if particle_index_1 == 0:
                            D += (self.G * J_X_second_derivative(XorY_1, XorY_2, X[0], Y[0], self.S[0]) * 
                            (J_X(X[1], Y[1], self.S[1]) + J_X(X[2], Y[2], self.S[2])))
                            D += (self.J * J_Z_second_derivative(XorY_1, XorY_2, X[0], Y[0], self.S[0]) * 
                            (J_Z(X[1], Y[1], self.S[1]) + J_Z(X[2], Y[2], self.S[2])))
                    elif particle_index_1 == 1:
                            D += (self.G * J_X_second_derivative(XorY_1, XorY_2, X[1], Y[1], self.S[1]) * 
                            (J_X(X[0], Y[0], self.S[0]) + J_X(X[2], Y[2], self.S[2] + J_X(X[3], Y[3], self.S[3]))))
                            D += (self.J * J_Z_second_derivative(XorY_1, XorY_2, X[1], Y[1], self.S[1]) * 
                            (J_Z(X[0], Y[0], self.S[0]) + J_Z(X[2], Y[2], self.S[2])  + J_Z(X[3], Y[3], self.S[3])))
                    elif particle_index_1 == 2:
                            D += (self.G * J_X_second_derivative(XorY_1, XorY_2, X[2], Y[2], self.S[2]) * 
                            (J_X(X[0], Y[0], self.S[0]) + J_X(X[1], Y[1], self.S[1] + J_X(X[3], Y[3], self.S[3]))))
                            D += (self.J * J_Z_second_derivative(XorY_1, XorY_2, X[2], Y[2], self.S[2]) * 
                            (J_Z(X[0], Y[0], self.S[0]) + J_Z(X[1], Y[1], self.S[1])  + J_Z(X[3], Y[3], self.S[3])))
                    elif particle_index_1 == 3:
                            D += (self.G * J_X_second_derivative(XorY_1, XorY_2, X[3], Y[3], self.S[3]) * 
                            (J_X(X[1], Y[1], self.S[1]) + J_X(X[2], Y[2], self.S[2])))
                            D += (self.J * J_Z_second_derivative(XorY_1, XorY_2, X[3], Y[3], self.S[3]) * 
                            (J_Z(X[1], Y[1], self.S[1]) + J_Z(X[2], Y[2], self.S[2])))
                    return D
            else:
                    if particle_index_1 == 0:
                            if particle_index_2 == 1:
                                    D += (self.G * J_X_derivative(XorY_1, X[0], Y[0], self.S[0]) *
                                         J_X_derivative(XorY_2, X[1], Y[1], self.S[1]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[0], Y[0], self.S[0]) *
                                         J_Z_derivative(XorY_2, X[1], Y[1], self.S[1]))
                            elif particle_index_2 == 2:
                                    D += (self.G * J_X_derivative(XorY_1, X[0], Y[0], self.S[0]) *
                                         J_X_derivative(XorY_2, X[2], Y[2], self.S[2]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[0], Y[0], self.S[0]) *
                                         J_Z_derivative(XorY_2, X[2], Y[2], self.S[2]))
                            return D
                    elif particle_index_1 == 1:
                            if particle_index_2 == 0:
                                    D += (self.G * J_X_derivative(XorY_1, X[1], Y[1], self.S[1]) *
                                         J_X_derivative(XorY_2, X[0], Y[0], self.S[0]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[1], Y[1], self.S[1]) *
                                         J_Z_derivative(XorY_2, X[0], Y[0], self.S[0]))
                            elif particle_index_2 == 2:
                                    D += (self.G * J_X_derivative(XorY_1, X[1], Y[1], self.S[1]) *
                                         J_X_derivative(XorY_2, X[2], Y[2], self.S[2]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[1], Y[1], self.S[1]) *
                                         J_Z_derivative(XorY_2, X[2], Y[2], self.S[2]))
                            elif particle_index_2 == 3:
                                    D += (self.G * J_X_derivative(XorY_1, X[1], Y[1], self.S[1]) *
                                         J_X_derivative(XorY_2, X[3], Y[3], self.S[3]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[1], Y[1], self.S[1]) *
                                         J_Z_derivative(XorY_2, X[3], Y[3], self.S[3]))
                            return D
                    elif particle_index_1 == 2:
                            if particle_index_2 == 0:
                                    D += (self.G * J_X_derivative(XorY_1, X[2], Y[2], self.S[2]) *
                                         J_X_derivative(XorY_2, X[0], Y[0], self.S[0]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[2], Y[2], self.S[2]) *
                                         J_Z_derivative(XorY_2, X[0], Y[0], self.S[0]))
                            elif particle_index_2 == 1:
                                    D += (self.G * J_X_derivative(XorY_1, X[2], Y[2], self.S[2]) *
                                         J_X_derivative(XorY_2, X[1], Y[1], self.S[1]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[2], Y[2], self.S[2]) *
                                         J_Z_derivative(XorY_2, X[1], Y[1], self.S[1]))
                            elif particle_index_2 == 3:
                                    D += (self.G * J_X_derivative(XorY_1, X[2], Y[2], self.S[2]) *
                                         J_X_derivative(XorY_2, X[3], Y[3], self.S[3]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[2], Y[2], self.S[2]) *
                                         J_Z_derivative(XorY_2, X[3], Y[3], self.S[3]))
                            return D
                    elif particle_index_1 == 3:
                            if particle_index_2 == 1:
                                    D += (self.G * J_X_derivative(XorY_1, X[3], Y[3], self.S[3]) *
                                         J_X_derivative(XorY_2, X[1], Y[1], self.S[1]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[3], Y[3], self.S[3]) *
                                         J_Z_derivative(XorY_2, X[1], Y[1], self.S[1]))
                            elif particle_index_2 == 2:
                                    D += (self.G * J_X_derivative(XorY_1, X[3], Y[3], self.S[3]) *
                                         J_X_derivative(XorY_2, X[2], Y[2], self.S[2]))
                                    D += (self.J * J_Z_derivative(XorY_1, X[3], Y[3], self.S[3]) *
                                         J_Z_derivative(XorY_2, X[2], Y[2], self.S[2]))
                            return D
                        
#Implementation of the spin coherent state action for a spin system

class Spin_System:
    def __init__(self, num_particles, num_time_slices, spins, beta, hamiltonian, lambda_):
        self.H = hamiltonian
        self.N = num_particles
        self.T = num_time_slices
        self.S = spins
        self.beta = beta
        self.Lambda = lambda_

    # geometric phase of action
    def berry_phase(self, X, Y):
        bp = 0
        for i in range(self.N):
            for j in range(self.T-1):
                bp += -2j*self.S[i]*((X[i,j+1]-X[i,j])*Y[i,j] - (Y[i,j+1]-Y[i,j])*X[i,j])/(1+X[i,j]**2+Y[i,j]**2)
            bp += -2j*self.S[i]*((X[i,0]-X[i,-1])*Y[i,-1] - (Y[i,0]-Y[i,-1])*X[i,-1])/(1+X[i,-1]**2+Y[i,-1]**2)
        return bp

    # derivative with respect to x or y (depending on XorY) of the n_ind-th particle in the t_ind-th timeslice
    def berry_phase_derivative(self, XorY, n_ind, t_ind, X, Y):
        if t_ind == self.T-1:
            t_ind = -1
        der = 0
        if XorY == 0:
            der += -Y[n_ind, t_ind+1]/(1+X[n_ind, t_ind]**2+Y[n_ind, t_ind]**2)
            der += -2*X[n_ind, t_ind]*(((X[n_ind, t_ind+1]-X[n_ind, t_ind])*Y[n_ind, t_ind]
                                        - (Y[n_ind, t_ind+1]-Y[n_ind, t_ind])*X[n_ind, t_ind])
                                       /(1+X[n_ind, t_ind]**2+Y[n_ind, t_ind]**2)**2)
            der += (Y[n_ind, t_ind-1]/(1+X[n_ind, t_ind-1]**2+Y[n_ind, t_ind-1]**2))
            der = -2j*self.S[n_ind]*der
        else:
            der += X[n_ind, t_ind+1]/(1+X[n_ind, t_ind]**2+Y[n_ind, t_ind]**2)
            der += -2*Y[n_ind, t_ind]*(((X[n_ind, t_ind+1]-X[n_ind, t_ind])*Y[n_ind, t_ind]
                                             - (Y[n_ind, t_ind+1]-Y[n_ind, t_ind])*X[n_ind, t_ind])
                                            /(1+X[n_ind, t_ind]**2+Y[n_ind, t_ind]**2)**2)
            der += -X[n_ind, t_ind-1]/(1+X[n_ind, t_ind-1]**2+Y[n_ind, t_ind-1]**2)
            der = -2j*self.S[n_ind]*der
        return der

    # second derivatives work in the same way
    def berry_phase_second_derivative(self, XorY_1, XorY_2, number_index_1, number_index_2, time_index_1, time_index_2, X, Y):
        if number_index_1 != number_index_2: return 0
        else:
            index_1 = time_index_1
            index_2 = time_index_2
            if index_1 ==  self.T-1:
                index_1 = -1
            if index_2 == self.T-1:
                index_2 = -1
            if XorY_1 == 0:
                if XorY_2 == 0:
                    if index_1 == index_2:
                      der = 0
                      der += 4*X[number_index_1, index_1]*Y[number_index_1,index_1+1]/(1+X[number_index_1, index_1]**2+Y[number_index_1,index_1]**2)**2
                      der += (((X[number_index_1, index_1+1]-X[number_index_1,index_1])*Y[number_index_1,index_1]-(Y[number_index_1,index_1+1]-Y[number_index_1,index_1])*X[number_index_1,index_1])
                             *(8*X[number_index_1,index_1]**2/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**3
                               -2/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2))
                      der = 2j*self.S[number_index_1]*der
                      return der
                    elif (index_1 + 1)%self.T == index_2%self.T:
                        return 4j*self.S[number_index_1]*X[number_index_1,index_1]*Y[number_index_1,index_1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                    elif (index_2+1)%self.T == index_1%self.T:
                        return 4j*self.S[number_index_1]*X[number_index_1,index_2]*Y[number_index_1,index_2]/(1+X[number_index_1, index_2]**2+Y[number_index_1,index_2]**2)**2
                    else:
                        return 0
                else:
                    if index_1 == index_2: 
                        der = 0

                        der += -2*X[number_index_1,index_1]*X[number_index_1,index_1+1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                        der += 2*Y[number_index_1,index_1]*Y[number_index_1,index_1+1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                        der += 8*X[number_index_1,index_1]*Y[number_index_1,index_1]*(((X[number_index_1,index_1+1]-X[number_index_1,index_1])*Y[number_index_1,index_1]
                                                                                       - (Y[number_index_1,index_1+1]-Y[number_index_1,index_1])*X[number_index_1,index_1])
                                                                                      /(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**3) 
                        der *= -2j*self.S[number_index_1]
                        return der
                    elif (index_1 + 1)%self.T == index_2%self.T:
                        return -2j*self.S[number_index_1] * (2*X[number_index_1,index_1]**2/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                                                        - 1/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2))
                    elif (index_2 + 1)%self.T == index_1%self.T:
                        return -2j*self.S[number_index_1]*(-2*Y[number_index_1,index_2]**2/(1+X[number_index_1,index_2]**2+Y[number_index_1,index_2]**2)**2
                                                      + 1/(1+X[number_index_1,index_2]**2+Y[number_index_1,index_2]**2))
                    else:
                        return 0
            else:
                if XorY_2 == 0:
                    if index_1 == index_2: 
                        der = 0

                        der += -2*X[number_index_1,index_1]*X[number_index_1,index_1+1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                        der += 2*Y[number_index_1,index_1]*Y[number_index_1,index_1+1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                        der += 8*X[number_index_1,index_1]*Y[number_index_1,index_1]*(((X[number_index_1,index_1+1]-X[number_index_1,index_1])*Y[number_index_1,index_1] - (Y[number_index_1,index_1+1]-Y[number_index_1,index_1])
                                                         *X[number_index_1,index_1])/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**3)
                        der *= -2j*self.S[number_index_1]
                        return der
                    elif (index_1 + 1)%self.T == index_2%self.T:
                        return -2j*self.S[number_index_1]*(-2*Y[number_index_1,index_1]**2/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                                                      + 1/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2))
                    elif (index_2 + 1)%self.T == index_1%self.T:
                        return -2j*self.S[number_index_1] * (2*X[number_index_1,index_2]**2/(1+X[number_index_1,index_2]**2+Y[number_index_1,index_2]**2)**2
                                                        - 1/(1+X[number_index_1,index_2]**2+Y[number_index_1,index_2]**2))
                    else:
                        return 0
                else:
                    if index_1 == index_2:
                        der = 0
                        der += -4*X[number_index_1,index_1+1]*Y[number_index_1,index_1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                        der += ((8*Y[number_index_1,index_1]**2/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**3 - 2/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2)
                                *((X[number_index_1,index_1+1]-X[number_index_1,index_1])*Y[number_index_1,index_1]-(Y[number_index_1,index_1+1]-Y[number_index_1,index_1])*X[number_index_1,index_1]))
                        der *= -2j*self.S[number_index_1]
                        return der
                    elif (index_1 + 1)%self.T == index_2%self.T:
                        return -4j*self.S[number_index_1]*X[number_index_1,index_1]*Y[number_index_1,index_1]/(1+X[number_index_1,index_1]**2+Y[number_index_1,index_1]**2)**2
                    elif (index_2 + 1)%self.T == index_1%self.T:
                        return -4j*self.S[number_index_1]*X[number_index_1,index_2]*Y[number_index_1,index_2]/(1+X[number_index_1,index_2]**2+Y[number_index_1,index_2]**2)**2
                    else:
                        return 0

    # Gives the logarithm of the stereographic volume element
    def vol_log(self, x, y):
        return 2*np.log(x**2+y**2+1)
                
    def vol_log_derivative(self, XorY, x, y):
        if XorY == 0:
            return 4*x/(1+x**2+y**2)
        else:
            return 4*y/(1+x**2+y**2)

    def vol_log_second_derivative(self, XorY_1, XorY_2, x, y):
        if XorY_1 == 0:
            if XorY_2 == 0:
                return 4*(y**2-x**2-1)/(x**2+y**2+1)**2
            else:
                return -8*x*y/(x**2+y**2+1)**2
        else:
            if XorY_2 == 0:
                return -8*x*y/(x**2+y**2+1)**2
            else:
                return 4*(x**2-y**2-1)/(x**2+y**2+1)**2

    # Gives the total action (S' in our paper)
    def action(self, X, Y):
        vol_sum = 0
        ham_sum = 0
        for j in range(self.T):
            for i in range(self.N):
                vol_sum += self.vol_log(X[i, j], Y[i, j])
            ham_sum += self.H.energy(X[:, j], Y[:, j])
        return self.berry_phase(X, Y) + self.beta/self.T * ham_sum + vol_sum
    
    def action_derivative(self, XorY, particle_index, time_index, X, Y):
        return (self.berry_phase_derivative(XorY, particle_index, time_index, X, Y) + self.beta/self.T * self.H.derivative(XorY, particle_index, X[:, time_index], Y[:, time_index])
                + self.vol_log_derivative(XorY, X[particle_index,time_index], Y[particle_index,time_index]))
    
    def action_second_derivative(self, XorY_1, XorY_2, particle_index_1, particle_index_2, time_index_1, time_index_2, X, Y):
        second_deriv = 0
        if time_index_1 == time_index_2:
            if particle_index_1 == particle_index_2:
                second_deriv += self.vol_log_second_derivative(XorY_1, XorY_2, X[particle_index_1, time_index_1], Y[particle_index_1, time_index_1])
            second_deriv += self.beta/self.T * self.H.second_derivative(XorY_1, XorY_2, particle_index_1, particle_index_2, X[:, time_index_1], Y[:, time_index_1])
        second_deriv += self.berry_phase_second_derivative(XorY_1, XorY_2, particle_index_1, particle_index_2, time_index_1, time_index_2, X, Y)
        return second_deriv
    
    # Gives the action without the volume element (S in the paper)
    def bosonic_action(self, X, Y):
        ham_sum = 0
        for j in range(self.T):
            ham_sum += self.H.energy(X[:, j], Y[:, j])
        return self.berry_phase(X, Y) + self.beta/self.T * ham_sum

    def bosonic_action_derivative(self, XorY, particle_index, time_index, X, Y):
        return (self.berry_phase_derivative(XorY, particle_index, time_index, X, Y) + self.beta/self.T * self.H.derivative(XorY, particle_index, X[:, time_index], Y[:, time_index]))

    def hessian_matrix(self, X, Y):
        hess = np.zeros((2*self.N*self.T, 2*self.N*self.T), dtype = np.complex128)
        for i in range(self.N):
            for j in range(self.T):
                for k in range(self.N):
                    for l in range(self.T):
                        hess[self.N*j+i, self.N*l+k] = self.action_second_derivative(0, 0, i, k, j, l, X, Y)
                        hess[(self.N*self.T)+self.N*j+i, self.N*l+k] = self.action_second_derivative(1, 0, i, k, j, l, X, Y)
                        hess[self.N*j+i, (self.N*self.T)+self.N*l+k] = self.action_second_derivative(0, 1, i, k, j, l, X, Y)
                        hess[(self.N*self.T)+self.N*j+i, (self.N*self.T)+self.N*l+k] = self.action_second_derivative(1, 1, i, k, j, l, X, Y)
        return hess

    # Gives the steps in the non blow-up gradient flows
    def dzdt(self, XorY, X, Y):
        time_der = np.zeros((self.N, self.T), dtype = np.complex128)
        for i in range(self.N):
            for j in range(self.T):
                time_der[i, j] = np.conj(self.action_derivative(XorY, i, j, X, Y))
        time_der *= np.exp(-2*self.bosonic_action(X, Y).real/self.Lambda)
        return time_der
    
    def dJdt(self, X, Y, J_in):
        dJ = np.zeros((2*self.N*self.T, 2*self.N*self.T), dtype = np.complex128)
        aux_mat_1 = np.zeros((2*self.N*self.T, 2*self.N*self.T), dtype = np.complex128)
        for i in range(self.N):
            for j in range(self.T):
                for k in range(self.N):
                    for l in range(self.T):
                        aux_mat_1[self.N*j+i, self.N*l+k] += (self.action_derivative(0, i, j, X, Y)*(-self.bosonic_action_derivative(0, k, l, X, Y)/self.Lambda))
                        aux_mat_1[self.T*self.N+self.N*j+i, self.T*self.N+self.N*l+k] += (self.action_derivative(1, i, j, X, Y)*(-self.bosonic_action_derivative(1, k, l, X, Y)/self.Lambda))
                        aux_mat_1[self.N*j+i, self.T*self.N+self.N*l+k] += (self.action_derivative(0, i, j, X, Y)*(-self.bosonic_action_derivative(1, k, l, X, Y)/self.Lambda))
                        aux_mat_1[self.T*self.N+self.N*j+i, self.N*l+k] += (self.action_derivative(1, i, j, X, Y)*(-self.bosonic_action_derivative(0, k, l, X, Y)/self.Lambda))


        dJ += np.conj((self.hessian_matrix(X, Y) + aux_mat_1)@J_in)
        aux_mat_2 = np.zeros((2*self.N*self.T, 2*self.N*self.T), dtype = np.complex128)
        for i in range(self.N):
            for j in range(self.T):
                for k in range(self.N):
                    for l in range(self.T):
                        aux_mat_2[self.N*j+i, self.N*l+k] += (np.conj(self.action_derivative(0, i, j, X, Y))*(-self.bosonic_action_derivative(0, k, l, X, Y)/self.Lambda))
                        aux_mat_2[self.T*self.N+self.N*j+i, self.T*self.N+self.N*l+k] += (np.conj(self.action_derivative(1, i, j, X, Y))*(-self.bosonic_action_derivative(1, k, l, X, Y)/self.Lambda))
                        aux_mat_2[self.N*j+i, self.T*self.N+self.N*l+k] += (np.conj(self.action_derivative(0, i, j, X, Y))*(-self.bosonic_action_derivative(1, k, l, X, Y)/self.Lambda))
                        aux_mat_2[self.T*self.N+self.N*j+i, self.N*l+k] += (np.conj(self.action_derivative(1, i, j, X, Y))*(-self.bosonic_action_derivative(0, k, l, X, Y)/self.Lambda))
        dJ += aux_mat_2@J_in
        dJ *= np.exp(-2 * self.bosonic_action(X, Y).real/self.Lambda)
        return dJ


class Flow:
    # Class to implement flows
    def __init__(self, flow_time, flow_steps, spin_syst):
        self.syst = spin_syst
        self.flow_time = flow_time
        self.flow_steps = flow_steps

        
    def simple_step(self, X_in, Y_in, J_in, dt):
        X_out = X_in + dt * self.syst.dzdt(0, X_in, Y_in)
        Y_out = Y_in + dt * self.syst.dzdt(1, X_in, Y_in)
        J_out = J_in + dt * self.syst.dJdt(X_in, Y_in, J_in)
        return (X_out, Y_out, J_out)

    # Single step for 4th order runge-kutta
    def rk4_step(self, X_in, Y_in, J_in, dt):
        k1_x = self.syst.dzdt(0, X_in, Y_in)
        k1_y = self.syst.dzdt(1, X_in, Y_in)
        k1_J = self.syst.dJdt(X_in, Y_in, J_in)
       
        k2_x = self.syst.dzdt(0, X_in+dt/2*k1_x, Y_in+dt/2*k1_y)
        k2_y = self.syst.dzdt(1, X_in+dt/2*k1_x, Y_in+dt/2*k1_y)
        k2_J = self.syst.dJdt(X_in+dt/2*k1_x, Y_in+dt/2*k1_y, J_in+dt/2*k1_J)
       
        k3_x = self.syst.dzdt(0, X_in+dt/2*k2_x, Y_in+dt/2*k2_y)
        k3_y = self.syst.dzdt(1, X_in+dt/2*k2_x, Y_in+dt/2*k2_y)
        k3_J = self.syst.dJdt(X_in+dt/2*k2_x, Y_in+dt/2*k2_y, J_in+dt/2*k2_J)
       
        k4_x = self.syst.dzdt(0, X_in+dt*k3_x, Y_in+dt*k3_y)
        k4_y = self.syst.dzdt(1, X_in+dt*k3_x, Y_in+dt*k3_y)
        k4_J = self.syst.dJdt(X_in+dt*k3_x, Y_in+dt*k3_y, J_in+dt*k3_J)

        X_out = X_in + dt/6*(k1_x+2*k2_x+2*k3_x+k4_x)
        Y_out = Y_in + dt/6*(k1_y+2*k2_y+2*k3_y+k4_y)
        J_out = J_in + dt/6*(k1_J+2*k2_J+2*k3_J+k4_J)
        return (X_out, Y_out, J_out)


    # Adaptive flow, checking that flow is increasing the real part of its action and not drifting in imaginary part too much
    def adaptive_flow(self, X_in, Y_in):
        t_count = 0
        dt_base = self.flow_time/self.flow_steps
        dt = dt_base
        X = X_in
        Y = Y_in
        J = np.eye(2*self.syst.T*self.syst.N, 2*self.syst.T*self.syst.N)
        while t_count<self.flow_time:
            X_flow, Y_flow, J_flow = self.simple_step(X, Y, J, dt) #self.rk4_step(X, Y, J, dt)#
            if self.syst.action(X_flow, Y_flow).real >= self.syst.action(X, Y).real and np.abs(self.syst.action(X_flow, Y_flow).imag - self.syst.action(X, Y).imag)<0.2:
                X = X_flow
                Y = Y_flow
                J = J_flow
                t_count += dt
            elif dt_base/dt > 10**10:
                print("STUCK AT TIME {}".format(t_count))
                return (X, Y, J)
            else:
                print(self.syst.action(X, Y))
                print(self.syst.action(X_flow, Y_flow))
                print("Halving at beta = {}".format(self.syst.beta))
                dt = dt/2
        return (X, Y, J)

# Executes one QMC run at beta, all other parameters taken care of. Preconfigured for ease of parallelization
def full_qmc(beta):
    N = 3
    T = 3
    S = []
    for i in range(N):
        S.append(10)
    Lambda = 300*beta
    ham = Frustrated_Triplet_Hamiltonian(N, S, 1, 1) 
    syst = Spin_System(N, T, S, beta, ham, Lambda)
    flow = Flow(0.02/(beta**(0.75)), 100, syst) 
    base_flow = Flow(0, 100, syst)
##    X = np.random.normal(size = (N, T), scale = 0.01)
##    Y = np.random.normal(size = (N, T), loc = 1.0, scale = 0.01)
    # better starting point
    # effective action ~-180
    X = np.array([[-1.49905465+0.j, -1.49535116+0.j, -1.4916209 +0.j],
                  [ 0.05985293+0.j,  0.06081368+0.j,  0.0606531 +0.j],
                  [ 2.00016556+0.j,  2.00504165+0.j,  1.99667781+0.j]], dtype = np.complex128)
    Y = np.array([[ 1.86310496e-02+0.j,  1.65341043e-02+0.j,  1.40666676e-02+0.j],
                  [-8.50735076e-06+0.j,  1.97718627e-05+0.j, -2.57619016e-05+0.j],
                  [-4.19499942e-02+0.j, -3.11212969e-02+0.j, -3.82962198e-02+0.j]], dtype = np.complex128)

    num_samples = 1000
    num_thermalization = 1000

    expector = syst.H.energy
    drift_const = 0.08/np.sqrt(beta)
    base_inte, base_phase, base_acc = QMC(num_samples, num_thermalization, syst, base_flow, X, Y, expector, drift_const) 
    print("Value: {}".format(base_inte/base_phase))
    print("<sign>: {}".format(np.abs(base_phase/num_samples)))
    drift_const = 0.00004/(beta**(0.35)) 
    inte, phase, acc = QMC(num_samples, num_thermalization, syst, flow, X, Y, expector, drift_const) 
    return (inte/phase, np.abs(phase/num_samples), acc, base_inte/base_phase, np.abs(base_phase/num_samples), base_acc)

def main():
    # Performs 5 QMC runs in parallel over various values of beta.
    num_betas = 10
    num_samples = 5
    betas = []

    for i in range(num_betas):
        for j in range(num_samples):
            betas.append(1.0-i*0.1)

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(full_qmc, [beta for beta in betas])
    pool.close()
    print(betas)
    print(results)

# QMC with all parameters adjustable
# first performs num_thermalization thermalization steps, then num_samples actual samples
# syst and flow give the information of the actual system and flow, it starts at starting_X and starting_Y,
# takes the expectation of expector, and each step is normally distributed with standard deviation of drift_const
def QMC(num_samples, num_thermalization, syst, flow, starting_X, starting_Y, expector, drift_const):
    print("DRIFT CONSTANT FOR BETA={}: {}".format(syst.beta, drift_const))
    X = starting_X
    Y = starting_Y
    X_prime, Y_prime, J = flow.adaptive_flow(X, Y)
    accepted = 0
    eff_action = syst.action(X_prime, Y_prime) - np.log(np.linalg.det(J))
    integral = 0
    residual_phase = 0
    for i in range(num_samples+num_thermalization):
        print(">> {}".format(i))
        delta_X = np.random.normal(scale = drift_const, size = X.shape)
        delta_Y = np.random.normal(scale = drift_const, size = Y.shape)
        X_next = X + delta_X
        Y_next = Y + delta_Y
        X_next_prime, Y_next_prime, J_next = flow.adaptive_flow(X_next, Y_next)
        eff_action_next = syst.action(X_next_prime, Y_next_prime) - np.log(np.linalg.det(J_next))
        if np.random.uniform() < min(1, np.exp(-(eff_action_next.real - eff_action.real))):
            X = X_next
            Y = Y_next
            X_prime = X_next_prime
            Y_prime = Y_next_prime
            eff_action = eff_action_next
            accepted += 1
	    if i >= num_thermalization:
		for j in range(syst.T):
	    	    print(expector(X_prime[:, j], Y_prime[:, j]))
		    ham_avg += 1/syst.T*expector(X_prime[:, j], Y_prime[:, j])
		integral += ham_avg * np.exp(-1j * eff_action.imag) #Changing from row to column should fix
		residual_phase += np.exp(-1j * eff_action.imag)
    print("Number accepted: {}".format(accepted))
    return (integral, residual_phase, accepted)
    

if __name__ == "__main__":
    main()

