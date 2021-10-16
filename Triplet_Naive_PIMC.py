import numpy as np
from scipy.linalg import expm

# Triplet PIMC

S = [40, 40, 40]
T = 3

def Spin_X(S):
    J_X = np.zeros((int(2*S+1), int(2*S+1)), dtype = np.complex128)
    for i in range(int(2*S+1)):
        for j in range(int(2*S+1)):
            if i == j+1:
                J_X[i,j] = 0.5*np.sqrt((S+1)*(i+j+1)-(i+1)*(j+1))
            elif i+1 == j:
                J_X[i,j] = 0.5*np.sqrt((S+1)*(i+j+1)-(i+1)*(j+1))
    return J_X

def Spin_X_x_basis(S):
    return Spin_Z(S) 

def Spin_Z(S):
    J_Z = np.zeros((int(2*S+1), int(2*S+1)), dtype = np.complex128)
    for i in range(int(2*S+1)):
        J_Z[i, i] = -S + i
    return J_Z

def d_matrix(j, m, m_prime):
    su = 0
    s_min = int(max(0, m-m_prime))
    s_max = int(min(j+m, j-m_prime))
    for s in range(s_min, s_max+1):
        su += (-1)**(int(m_prime-m)+s)/(np.math.factorial(int(j+m-s))*np.math.factorial(int(s))
                                        *np.math.factorial(int(m_prime-m+s))
                                        *np.math.factorial(int(j-m_prime-s)))
    su *= np.sqrt(float(np.math.factorial(int(j+m_prime))*np.math.factorial(int(j-m_prime))
                        *np.math.factorial(int(j+m))*np.math.factorial(int(j-m))))*(np.sqrt(2)/2)**(2*j)
    return su

def x_eigenstate(j, m):
    vect = np.zeros(int(2*j+1), dtype = np.complex128)
    for i in range(int(2*j+1)):
        m_prime = -j + i
        vect[i] = d_matrix(j, m, m_prime)
    return vect



def hamiltonian(S, triplet_state_1, triplet_state_2, J, Gamma):
    # Calculate <triplet_state_1| H |triplet_state_2>
    amplitude = 0
    spin_Z = Spin_Z(S)
    spin_X = Spin_X(S)

    amplitude += J*(((state(triplet_state_1[0],S).T@spin_Z@state(triplet_state_2[0], S))[0,0])
                    *((state(triplet_state_1[1], S).T@spin_Z@state(triplet_state_2[1], S))[0,0])
                    *((state(triplet_state_1[2], S).T@state(triplet_state_2[2], S))[0,0]))

    amplitude += J*(((state(triplet_state_1[1],S).T@spin_Z@state(triplet_state_2[1], S))[0,0])
                    *((state(triplet_state_1[2], S).T@spin_Z@state(triplet_state_2[2], S))[0,0])
                    *((state(triplet_state_1[0], S).T@state(triplet_state_2[0], S))[0,0]))

    amplitude += J*(((state(triplet_state_1[0],S).T@spin_Z@state(triplet_state_2[0], S))[0,0])
                    *((state(triplet_state_1[2], S).T@spin_Z@state(triplet_state_2[2], S))[0,0])
                    *((state(triplet_state_1[1], S).T@state(triplet_state_2[1], S))[0,0]))

    amplitude += Gamma*((state(triplet_state_1[0], S).T@spin_X@state(triplet_state_2[0], S))[0,0]
                    *(state(triplet_state_1[1], S).T@spin_X@state(triplet_state_2[1], S))[0,0]
                    *(state(triplet_state_1[2], S).T@state(triplet_state_2[2], S))[0,0])
    
    amplitude += Gamma*((state(triplet_state_1[1], S).T@spin_X@state(triplet_state_2[1], S))[0,0]
                    *(state(triplet_state_1[2], S).T@spin_X@state(triplet_state_2[2], S))[0,0]
                    *(state(triplet_state_1[0], S).T@state(triplet_state_2[0], S))[0,0])
    
    amplitude += Gamma*((state(triplet_state_1[0], S).T@spin_X@state(triplet_state_2[0], S))[0,0]
                    *(state(triplet_state_1[2], S).T@spin_X@state(triplet_state_2[2], S))[0,0]
                    *(state(triplet_state_1[1], S).T@state(triplet_state_2[1], S))[0,0])
    print("Hamiltonian matrix element: {}".format(amplitude))
    return amplitude

def single_boltzmann_amplitude(S, triplet_index_1_z, triplet_index_1_x, triplet_index_2_z, J, Gamma, beta, T):
    toreturn = 1
    toreturn *= np.exp(-J*beta*(spin_val(triplet_index_1_z[0], S)*spin_val(triplet_index_1_z[1], S)+
                              spin_val(triplet_index_1_z[1], S)*spin_val(triplet_index_1_z[2], S)+
                              spin_val(triplet_index_1_z[2], S)*spin_val(triplet_index_1_z[0], S)))
    toreturn *= (d_matrix(S, spin_val(triplet_index_1_x[0], S), spin_val(triplet_index_1_z[0], S))
                 *d_matrix(S, spin_val(triplet_index_1_x[1], S), spin_val(triplet_index_1_z[1], S))
                 *d_matrix(S, spin_val(triplet_index_1_x[2], S), spin_val(triplet_index_1_z[2], S)))
    toreturn *= np.exp(-Gamma*beta*(spin_val(triplet_index_1_x[0], S)*spin_val(triplet_index_1_x[1], S)+
                              spin_val(triplet_index_1_x[1], S)*spin_val(triplet_index_1_x[2], S)+
                              spin_val(triplet_index_1_x[2], S)*spin_val(triplet_index_1_x[0], S)))
    toreturn *= (d_matrix(S, spin_val(triplet_index_1_x[0], S), spin_val(triplet_index_2_z[0], S))
                 *d_matrix(S, spin_val(triplet_index_1_x[1], S), spin_val(triplet_index_2_z[1], S))
                 *d_matrix(S, spin_val(triplet_index_1_x[2], S), spin_val(triplet_index_2_z[2], S)))
    return toreturn


def total_boltzmann_amplitude(S, triplet_indices_z, triplet_indices_x, J, Gamma, beta, T):
    amp = 1
    for i in range(len(triplet_indices_z[0])-1):
        amp *= single_boltzmann_amplitude(S, triplet_indices_z[i], triplet_indices_x[i], triplet_indices_z[i+1], J, Gamma, beta, T)
    amp *= single_boltzmann_amplitude(S, triplet_indices_z[-1], triplet_indices_x[-1], triplet_indices_z[0], J, Gamma, beta, T)
    return amp

def state(i, S):
    s = np.zeros((int(2*S+1), 1), dtype = np.complex128)
    s[i] = 1
    return s

def spin_val(i, S):
    return i-S

    

print(np.math.factorial(3))
v = x_eigenstate(40, -3)
print(np.linalg.norm(v))


S = 10

print(Spin_X(S))
print(Spin_Z(S))

triplet_path_z = [[np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)]]
triplet_path_x = [[np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)]]

def PIMC(S, T, beta, thermalization, sweeps, J, Gamma):
    triplet_path_z = [[np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                      [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                      [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)]]
    triplet_path_x = [[np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                      [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)],
                      [np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1), np.random.randint(low = 0, high = 2*S+1)]]
    bolt = total_boltzmann_amplitude(S, triplet_path_z, triplet_path_x, J, Gamma, beta, T)
    for i in range(thermalization):
        print(i)
        for j in range(T):
            for k in range(T):
                path_proposal_z = triplet_path_z.copy()
                path_proposal_x = triplet_path_x.copy()
                path_proposal_z[j][k] = np.random.randint(low = 0, high = 2*S+1)
                path_proposal_x[j][k] = np.random.randint(low = 0, high = 2*S+1)
                if np.random.uniform() < (np.abs(total_boltzmann_amplitude(S, path_proposal_z, path_proposal_x, J, Gamma, beta, T))
                                          /np.abs(total_boltzmann_amplitude(S, triplet_path_z, triplet_path_x, J, Gamma, beta, T))):
                    triplet_path_z = path_proposal_z
                    triplet_path_x = path_proposal_x
                
    total_phase  = 0
    total_energy = 0
    for i in range(sweeps):
        print(i)
        for j in range(T):
            for k in range(T):
                path_proposal_z = triplet_path_z.copy()
                path_proposal_x = triplet_path_x.copy()
                path_proposal_z[j][k] = np.random.randint(low = 0, high = 2*S+1)
                path_proposal_x[j][k] = np.random.randint(low = 0, high = 2*S+1)
                if np.random.uniform() < (np.abs(total_boltzmann_amplitude(S, path_proposal_z, path_proposal_x, J, Gamma, beta, T))
                                          /np.abs(total_boltzmann_amplitude(S, triplet_path_z, triplet_path_x, J, Gamma, beta, T))):
                    triplet_path_z = path_proposal_z
                    triplet_path_x = path_proposal_x
        total_phase += np.exp(1j*np.angle(total_boltzmann_amplitude(S, triplet_path_z, triplet_path_x, J, Gamma, beta, T)))
        total_energy += hamiltonian(S, triplet_path_z[0], triplet_path_z[0], J, Gamma)
    return (total_phase/sweeps, total_energy/sweeps)



print(PIMC(10, 3, 1, 1000, 1000, 1, 1))


    

    
    
