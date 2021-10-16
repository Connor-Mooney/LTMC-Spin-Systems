import numpy as np

class Spin_System:
    def __init__(self, S, beta, T, h, h_deriv, second_h_deriv, L_B, L_F):
        self.S = S
        self.beta = beta
        self.T = T
        self.hamiltonian = (lambda x, y: h(x,y,S))
        self.hamiltonian_derivative = (lambda XorY, x, y: h_deriv(XorY, x, y, S))
        self.second_hamiltonian_derivative = (lambda XorY_1, XorY_2, x, y: second_h_deriv(XorY_1, XorY_2, x, y, S))
        self.Lambda_B = L_B
        self.Lambda_F = L_F

    def new_flow(self, X, Y, dt):
        # Steps in flow
        X_step = np.zeros(self.T, dtype = np.complex128)
        Y_step = np.zeros(self.T, dtype = np.complex128)
        Lambda_B = self.Lambda_B
        Lambda_F = self.Lambda_F
        for i in range(self.T):
            X_step[i] += (np.exp(-2 * self.bosonic_action(X, Y).real/Lambda_B)
                       * (np.conj(self.action_derivative(0, i, X, Y))))*dt
            Y_step[i] += (np.exp(-2 * self.bosonic_action(X, Y).real/Lambda_B)
                       * (np.conj(self.action_derivative(1, i, X, Y))))*dt 
        for i in range(self.T):
            X_step[i] += X[i]
            Y_step[i] += Y[i]
        return (X_step, Y_step)

    def new_jacobian_flow(self, J_in, X, Y, dt):
        # Step of flow for jacobian
        Lambda_B = self.Lambda_B
        Lambda_F = self.Lambda_F

        dJ = np.zeros((2*self.T, 2*self.T), dtype = np.complex128)
        aux_mat_1 = np.zeros((2*self.T, 2*self.T), dtype = np.complex128)
        for i in range(self.T):
            for j in range(self.T):
                aux_mat_1[i, j] += (self.action_derivative(0, i, X, Y)
                                    * (-self.bosonic_action_derivative(0, j, X, Y)/Lambda_B))
                aux_mat_1[self.T + i, self. T + j] += (self.action_derivative(1, i, X, Y)
                                                       * (-self.bosonic_action_derivative(1, j, X, Y)/Lambda_B))
                aux_mat_1[i, self. T + j] += (self.action_derivative(0, i, X, Y)
                                               * (-self.bosonic_action_derivative(1, j, X, Y)/Lambda_B))
                aux_mat_1[self. T + i, j] += (self.action_derivative(1, i, X, Y)
                                               * (-self.bosonic_action_derivative(0, j, X, Y)/Lambda_B))

        dJ += np.conj((self.hessian(X, Y) + aux_mat_1)@J_in)
        aux_mat_2 = np.zeros((2*self.T, 2*self.T), dtype = np.complex128)
        for i in range(self.T):
            for j in range(self.T):
                aux_mat_2[i, j] += (np.conj(self.action_derivative(0, i, X, Y))
                                    * (-self.bosonic_action_derivative(0, j, X, Y)/Lambda_B))
                aux_mat_2[self.T + i, self. T + j] += (np.conj(self.action_derivative(1, i, X, Y))
                                                       * (-self.bosonic_action_derivative(1, j, X, Y)/Lambda_B))
                aux_mat_2[i, self. T + j] += (np.conj(self.action_derivative(0, i, X, Y))
                                               * (-self.bosonic_action_derivative(1, j, X, Y)/Lambda_B))
                aux_mat_2[self. T + i, j] += (np.conj(self.action_derivative(1, i, X, Y))
                                               * (-self.bosonic_action_derivative(0, j, X, Y)/Lambda_B))
        dJ += aux_mat_2@J_in
        dJ *= np.exp(-2 * self.bosonic_action(X, Y).real/Lambda_B)
        dJ *= dt
        return J_in + dJ

    
    def vol(self, X, Y):
        # Volume form
        v = 1
        for i in range(self.T):
            v *= 1/(1+X[i]**2+Y[i]**2)**2
        return v

    def vol_derivative(self, XorY, index, X, Y):
        # Derivative of volume form
        v = self.vol(X, Y) * 1/(X[index]**2+Y[index]**2+1)**2 * (-4)
        if XorY == 0:
            return v*X[index]
        else:
            return v*Y[index]

    def vol_log(self, x, y):
        # log(vol form)
        return 2*np.log(x**2+y**2+1)

    def vol_log_derivative(self, XorY, x, y):
        # Derivative of log(vol form)
        if XorY == 0:
            return 4*x/(x**2+y**2+1)
        else:
            return 4*y/(x**2+y**2+1)

    def vol_log_second_derivative(self, XorY_1, XorY_2, x, y):
        # Second derivative of log(vol form)
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


    def berry_phase(self, X, Y):
        bp = 0
        for i in range(self.T-1):
            bp += ((X[i+1]-X[i])*Y[i] - (Y[i+1]-Y[i])*X[i])/(1+X[i]**2+Y[i]**2)

        #Enforcing periodicity because we're taking a trace
        bp += ((X[0]-X[-1])*Y[-1] - (Y[0]-Y[-1])*X[-1])/(1+X[-1]**2+Y[-1]**2)
        bp = -2j*self.S*bp
        return bp

    def action(self, X,Y):
        vol_int = 0
        ham_int = 0
        for i in range(self.T):
            vol_int += self.vol_log(X[i], Y[i])
            ham_int += self.beta/self.T * self.hamiltonian(X[i], Y[i])
        return self.berry_phase(X, Y) + ham_int + vol_int

    def bosonic_action(self, X, Y):
        ham_int = 0
        for i in range(self.T):
            ham_int += self.beta/self.T * self.hamiltonian(X[i], Y[i])
        return self.berry_phase(X, Y) + ham_int

    def bosonic_action_derivative(self, XorY, index, X, Y):
        return self.berry_phase_derivative(XorY, index, X, Y) + self.beta/self.T * self.hamiltonian_derivative(XorY, X[index], Y[index])


    def berry_phase_derivative(self, XorY, index, X, Y):
        if index == self.T-1:
            index = -1
        der = 0
        if XorY == 0:
            der += -Y[index+1]/(1+X[index]**2+Y[index]**2)
            der += -2*X[index]*((X[index+1]-X[index])*Y[index] - (Y[index+1]-Y[index])*X[index])/(1+X[index]**2+Y[index]**2)**2
            der += Y[index-1]/(1+X[index-1]**2+Y[index-1]**2)
            der = -2j*self.S*der
        else:
            der += X[index+1]/(1+X[index]**2+Y[index]**2)
            der += -2*Y[index]*((X[index+1]-X[index])*Y[index] - (Y[index+1]-Y[index])*X[index])/(1+X[index]**2+Y[index]**2)**2
            der += -X[index-1]/(1+X[index-1]**2+Y[index-1]**2)
            der = -2j*self.S*der

        return der

    def action_derivative(self, XorY, index, X, Y):
        return self.berry_phase_derivative(XorY, index, X, Y) + self.beta/self.T * self.hamiltonian_derivative(XorY, X[index], Y[index]) \
               + self.vol_log_derivative(XorY, X[index], Y[index])

    def flow_step(self, X, Y, dt):
        X_step = np.zeros(self.T, dtype = np.complex128)
        Y_step = np.zeros(self.T, dtype = np.complex128)
        for i in range(self.T):
            X_step[i] = X[i] + np.conj(self.action_derivative(0, i, X, Y))*dt
            Y_step[i] = Y[i] + np.conj(self.action_derivative(1, i, X, Y))*dt
        return (X_step, Y_step)

    def berry_phase_second_derivative(self, XorY_1, XorY_2, index_1, index_2, X, Y):
        if index_1 == self.T-1:
            index_1 = -1
        if index_2 == self.T-1:
            index_2 = -1
        der = 0
        if XorY_1 == 0:
            if XorY_2 == 0:
                if index_1 == index_2:
                  der = 0
                  der += 4*X[index_1]*Y[index_1+1]/(1+X[index_1]**2+Y[index_1]**2)**2
                  der += (((X[index_1+1]-X[index_1])*Y[index_1]-(Y[index_1+1]-Y[index_1])*X[index_1])
                         *(8*X[index_1]**2/(1+X[index_1]**2+Y[index_1]**2)**3
                           -2/(1+X[index_1]**2+Y[index_1]**2)**2))
                  der = 2j*self.S*der
                  return der
                elif (index_1 + 1)%self.T == index_2%self.T:
                    return 4j*self.S*X[index_1]*Y[index_1]/(1+X[index_1]**2+Y[index_1]**2)**2
                elif (index_2+1)%self.T == index_1%self.T:
                    return 4j*self.S*X[index_2]*Y[index_2]/(1+X[index_2]**2+Y[index_2]**2)**2
                else:
                    return 0
            else:
                if index_1 == index_2: 
                    der = 0

                    der += -2*X[index_1]*X[index_1+1]/(1+X[index_1]**2+Y[index_1]**2)**2
                    der += 2*Y[index_1]*Y[index_1+1]/(1+X[index_1]**2+Y[index_1]**2)**2
                    der += 8*X[index_1]*Y[index_1]*(((X[index_1+1]-X[index_1])*Y[index_1] - (Y[index_1+1]-Y[index_1])*X[index_1])
                                                    /(1+X[index_1]**2+Y[index_1]**2)**3) 
                    der *= -2j*self.S
                    return der
                elif (index_1 + 1)%self.T == index_2%self.T:
                    return -2j*self.S * (2*X[index_1]**2/(1+X[index_1]**2+Y[index_1]**2)**2 - 1/(1+X[index_1]**2+Y[index_1]**2))
                elif (index_2 + 1)%self.T == index_1%self.T:
                    return -2j*self.S*(-2*Y[index_2]**2/(1+X[index_2]**2+Y[index_2]**2)**2 + 1/(1+X[index_2]**2+Y[index_2]**2))
                else:
                    return 0
        else:
            if XorY_2 == 0:
                if index_1 == index_2: 
                    der = 0

                    der += -2*X[index_1]*X[index_1+1]/(1+X[index_1]**2+Y[index_1]**2)**2
                    der += 2*Y[index_1]*Y[index_1+1]/(1+X[index_1]**2+Y[index_1]**2)**2
                    der += 8*X[index_1]*Y[index_1]*(((X[index_1+1]-X[index_1])*Y[index_1] - (Y[index_1+1]-Y[index_1])*X[index_1])
                                                    /(1+X[index_1]**2+Y[index_1]**2)**3)
                    der *= -2j*self.S
                    return der
                elif (index_1 + 1)%self.T == index_2%self.T:
                    return -2j*self.S*(-2*Y[index_1]**2/(1+X[index_1]**2+Y[index_1]**2)**2 + 1/(1+X[index_1]**2+Y[index_1]**2))
                elif (index_2 + 1)%self.T == index_1%self.T:
                    return -2j*self.S * (2*X[index_2]**2/(1+X[index_2]**2+Y[index_2]**2)**2 - 1/(1+X[index_2]**2+Y[index_2]**2))
                else:
                    return 0
            else:
                if index_1 == index_2:
                    der = 0
                    der += -4*X[index_1+1]*Y[index_1]/(1+X[index_1]**2+Y[index_1]**2)**2
                    der += ((8*Y[index_1]**2/(1+X[index_1]**2+Y[index_1]**2)**3 - 2/(1+X[index_1]**2+Y[index_1]**2)**2)
                            *((X[index_1+1]-X[index_1])*Y[index_1]-(Y[index_1+1]-Y[index_1])*X[index_1]))
                    der *= -2j*self.S
                    return der
                elif (index_1 + 1)%self.T == index_2%self.T:
                    return -4j*self.S*X[index_1]*Y[index_1]/(1+X[index_1]**2+Y[index_1]**2)**2
                elif (index_2 + 1)%self.T == index_1%self.T:
                    return -4j*self.S*X[index_2]*Y[index_2]/(1+X[index_2]**2+Y[index_2]**2)**2
                else:
                    return 0

    def action_second_derivative(self, XorY_1, XorY_2, index_1, index_2, X, Y):
        if index_1 == index_2:
            return self.berry_phase_second_derivative(XorY_1, XorY_2, index_1, index_2, X, Y) \
                   + self.beta/self.T * self.second_hamiltonian_derivative(XorY_1, XorY_2, X[index_1], Y[index_1])\
                   + self.vol_log_second_derivative(XorY_1, XorY_2, X[index_1], Y[index_1])
        else:
            return self.berry_phase_second_derivative(XorY_1, XorY_2, index_1, index_2, X, Y)

    def hessian(self, X, Y):
        H = np.zeros((2*self.T, 2*self.T), dtype = np.complex128)
        for i in range(self.T):
            for j in range(self.T):
               H[i, j] = self.action_second_derivative(0, 0, i, j, X, Y)
               H[i+self.T, j] = self.action_second_derivative(1, 0, i, j, X, Y)
               H[i, j+self.T] = self.action_second_derivative(0, 1, i, j, X, Y)
               H[i+self.T, j+self.T] = self.action_second_derivative(1, 1, i, j, X, Y)
        np.set_printoptions(linewidth=np.nan)
        return H

    def crit_point_weight(self, X, Y):
        derivVect = np.zeros(2*self.T, dtype = np.complex128)
        for i in range(self.T):
            derivVect[i] = self.action_derivative(0, i, X, Y)
            derivVect[self.T+i] = self.action_derivative(1, i, X, Y)
        return np.linalg.norm(derivVect)**2


    def jacobian_flow_step(self, J_in, X, Y, dt):
        """One step in antiholomorphic flow for the jacobian"""
        # Probably VERY slow, but it does the job for now
        H = self.hessian(X, Y)
        return J_in + np.conj(H.dot(J_in))*dt

        
class Flow_Implementation:
    def __init__(self, spin_syst):
        self.flow_iteration = spin_syst.new_flow
        self.j_iteration = spin_syst.new_jacobian_flow
        self.morse_function = spin_syst.action
        self.T = spin_syst.T
        self.S = spin_syst.S
        self.beta = spin_syst.beta

    def adaptive_method(self, X_start, Y_start, dt_base, t_flow):
        # Adaptive flow, making step size smaller if the holomorphic flow doesn't behave
        X = X_start
        Y = Y_start
        t_curr = 0
        dt = dt_base
        J = np.eye(2*self.T)
        reset_flag = True
        act_start = self.morse_function(X,Y)
        act = self.morse_function(X, Y)
        while t_curr < t_flow:
            if reset_flag:
                dt = dt_base
            J_flow = self.j_iteration(J, X, Y, dt)
            X_flow, Y_flow = self.flow_iteration(X, Y, dt)
            act_flow = self.morse_function(X_flow, Y_flow)
            if dt_base/dt > 10**10: #Cutoff is arbitrary, could change if wanted
                print("Stuck at time: {}".format(t_curr))
                print(act)
                print(act_flow)
                print("Squares: {}".format([(X[i]**2+Y[i]**2) for i in range(self.T)]))
                return (X, Y, J)

            elif act_flow.real - act.real>= 0  and np.abs(act_flow.imag - act.imag) < 0.1:
                X = X_flow
                Y = Y_flow
                J = J_flow
                act = act_flow
                t_curr += dt
                reset_flag = True
            else:
                dt = dt/2
                reset_flag = False
        return (X, Y, J)

    
def main():
    # Hamiltonian and derivatives
    def ham(x, y, S):
        return 2* S * y/(1+x**2+y**2)

    def ham_der(XorY, x, y, S):
        if XorY == 0:
            return -4*S*x*y/(x**2+y**2+1)**2
        else:
            return 2*S*(x**2-y**2+1)/(x**2+y**2+1)**2

    def ham_second_der(XorY_1, XorY_2, x, y, S):
        if XorY_1 == 0:
            if XorY_2 == 0:
                return -4*S*y*(-3*x**2+y**2+1)/(x**2+y**2+1)**3
            else:
                return -4*S*x*(x**2-3*y**2+1)/(x**2+y**2+1)**3     
        else:
            if XorY_2 == 0:
                return -4*x*S*(x**2-3*y**2+1)/(x**2+y**2+1)**3
            else:
                return 4*S*y*(-3*x**2+y**2-3)/(x**2+y**2+1)**3
    # Parameter values
    S = 40
    beta = 0.1
    T = 3
    Lambda_B = 80
    Lambda_F = 3
    flow_time = 0.01
    num_steps = 1000

    # Setting up variables
    syst = Spin_System(S, beta, T, ham, ham_der, ham_second_der, Lambda_B, Lambda_F)
    flower = Flow_Implementation(syst)
    X = np.zeros(syst.T, dtype = np.complex128)
    Y = np.zeros(syst.T, dtype = np.complex128)
    J = np.eye(2*syst.T)
    for i in range(syst.T):
        X[i] += np.random.normal(scale = 0.1, loc = 0.0)
        Y[i] += np.random.normal(scale = 0.1, loc = 0.0)
    np.set_printoptions(linewidth=np.inf)

    # Setting up lists to store data
    betas = np.zeros(10)
    error_bars_phase = np.zeros(10)
    avgs_phase = np.zeros(10)
    base_error_phase = np.zeros(10)
    base_avgs_phase = np.zeros(10)
    
    error_bars_energy = np.zeros(10)
    avgs_energy = np.zeros(10)
    base_error_energy = np.zeros(10)
    base_avgs_energy = np.zeros(10)

    direct_calculation = np.zeros(10)

    # Running QMC
    for i in range(10):
        beta = 1 - i*0.1
        betas[i] = beta
        syst.beta = beta
        baseline_energy = np.zeros(5, dtype = np.complex128)
        baseline_phase = np.zeros(5, dtype = np.complex128)
        exp_phase = np.zeros(5, dtype = np.complex128)
        exp_energy = np.zeros(5, dtype = np.complex128)

        for j in range(5):
            print("BETA: {}".format(beta))
            X = np.zeros(syst.T, dtype = np.complex128)
            Y = np.zeros(syst.T, dtype = np.complex128)
            for k in range(syst.T):
                X[k] += np.random.normal(scale = 0.1, loc = 0.0)
                Y[k] += np.random.normal(scale = 0.1, loc = -1.0)
            print("Baseline**********")
            inte, phase = quantum_monte_carlo(flower, syst.hamiltonian, num_steps, 0.3/np.sqrt(beta), X, Y, 0) #Need to tune drifts!
            baseline_energy[j] = inte/phase
            print("Value: {}".format(inte/phase))
            baseline_phase[j] = np.abs(phase/num_steps)
            print("<sign>: {}".format(np.abs(phase/num_steps)))
            print("******************")
            print("Actual************")
            inte, phase = quantum_monte_carlo(flower, syst.hamiltonian, num_steps, 0.03/np.sqrt(beta), X, Y, flow_time) #Need to tune drifts!
            exp_energy[j] = inte/phase
            exp_phase[j] = np.abs(phase/num_steps)
            print("Value: {}".format(inte/phase))
            print("<sign>: {}".format(np.abs(phase/num_steps)))

            print("******************")

        error_bars_phase[i] = np.std(exp_phase)
        avgs_phase[i] = np.mean(exp_phase)
        error_bars_energy[i] = np.std(exp_energy)
        avgs_energy[i] = np.mean(exp_energy)
        base_error_energy[i] = np.std(baseline_energy)
        base_avgs_energy[i] = np.mean(baseline_energy)
        base_error_phase[i] = np.std(baseline_phase)
        base_avgs_phase[i] = np.mean(baseline_phase)
        
    # Printing results
    print("From beta = 1, to beta = 0.1 in descending increments of 0.1")
    print("Error bars phase: {}".format(error_bars_phase))
    print("Baseline: {}".format(base_error_phase))
    print("Error bars energy: {}".format(error_bars_energy))
    print("Baseline: {}".format(base_error_energy))
    print("Average phase: {}".format(avgs_phase))
    print("Baseline: {}".format(base_avgs_phase))
    print("Average energy: {}".format(avgs_energy))
    print("Baseline: {}".format(base_avgs_energy))
    print(" ")
    print("Direct energy calculation: {}".format(direct_calc(S, beta)))

def quantum_monte_carlo(flow, expectation, num_proposals, drift_const, starting_X, starting_Y, flow_duration):
    #Set-up
    X = starting_X
    Y = starting_Y
    flow_time = flow_duration
    flow_step = flow_time/100
    X_prime, Y_prime, J = flow.adaptive_method(X, Y, flow_step, flow_time)
    print(X_prime)
    print(Y_prime)
    action = flow.morse_function
    expector = (lambda X, Y: off_diagonal_expansion(X, Y, flow.S, flow.beta, flow.T)) 
    print(action(X, Y))
    print("Hamiltonian: {}".format(expectation(X_prime[0], Y_prime[0])))

    print(action(X_prime, Y_prime))
    eff_act = action(X_prime, Y_prime) - np.log(np.linalg.det(J))
    print(eff_act)
    
    integral = 0
    residual_phase = 0
    accepted = 0
    for i in range(num_proposals+1000): # Main QMC iteration
        if i%200 == 0: # Printing out intermediary values
            print(i)
            if i>1000:
                print(">> value: {}".format(integral/residual_phase))
                print(">> <sign>: {}".format(np.abs(residual_phase/(i-1000))))
                print(">> Accepted: {}/{}".format(accepted, i))

        # Generating proposal
        deltaX = np.random.normal(scale = drift_const, size = flow.T)
        deltaY = np.random.normal(scale = drift_const, size = flow.T)
        X_next = X + deltaX
        Y_next = Y + deltaY
        X_next_prime, Y_next_prime, J_next = flow.adaptive_method(X_next, Y_next, flow_step, flow_time)
        eff_act_next = action(X_next_prime, Y_next_prime) - np.log(np.linalg.det(J_next))

        if np.random.uniform() < min(1, np.exp(-(eff_act_next.real - eff_act.real))): # Accept/Reject step

            X = X_next
            Y = Y_next
            X_prime = X_next_prime
            Y_prime = Y_next_prime
            eff_act = eff_act_next
            accepted += 1

        if i > 1000: # Adding values to phase/energy once past thermalization
            integral += expector(X_prime, Y_prime) * np.exp(-1j * eff_act.imag)
            residual_phase += np.exp(-1j * eff_act.imag)
    print("Accepted: {}".format(accepted))

    return (integral, residual_phase)

def off_diagonal_expansion(X, Y, S, beta, T):
    # Gives the value of <z_0| H e^(-beta*H/T) | Z_(T-1)>/<z_0| e^(-beta*H/T)|z_(T-1)>
    trig_arg = beta/(2*T)
    diff = (Y[0]+Y[-1]) + 1j*(X[0]-X[-1]) # z_0*-z_(T-1)
    prod = X[0]*X[-1] + Y[0]*Y[-1] + 1j*(X[0]*Y[-1] - X[-1]*Y[0]) #(z_0^*) * (z_(T-1))
    return S*((np.cosh(trig_arg)*diff - np.sinh(trig_arg)*(1+prod))/(np.cosh(trig_arg)*(1+prod) - np.sinh(trig_arg)*diff))

def direct_calc(S, beta):
    # Direct calcalculation of expected energy
    return (-np.sinh(beta*S)-S*np.sinh(beta*S)+S*np.sinh(beta*(S+1)))/(np.cosh(beta*S)-np.cosh(beta*(1+S)))

if __name__ == "__main__":
    main()
