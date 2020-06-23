import pandas as pd
import numpy as np
from scipy.stats import norm


class HiddenMarkovModelResults:

    def __init__(self, n_states, transition_matrix, state_features, fitted_values, forward_probabilities, scale_value):
        self.n_states = n_states
        self.transition_matrix = transition_matrix
        self.state_features = state_features
        self.fitted_values = fitted_values
        self.forward_probabilities = forward_probabilities
        self.scale_value = scale_value

    def forecast(self, steps):
        """
        This function predicts the subsequent value using a simluation method.
        For each timestep, the model simulates 1,000 subsequent observations
        derived from the state_features and the transition matrix

        Args:
            - steps: int, the number of out of box predictions to make.The OOB predictions become increasingly unstable as steps grows.
        Returns:
            - predicted_values: list, the OOB predictions
        """

        # create local variables
        transition_matrix = self.transition_matrix
        state_features = self.state_features

        print('Creating OOB predictions for HMM with transition matrix: ')
        print(transition_matrix)
        print('and state features: ')
        print(state_features)

        predicted_values = []
        s_1_probabilities = self.forward_probabilities[-1]
        for i in range(steps):
            s_2_probabilities = sum(np.multiply(s_1_probabilities, transition_matrix.T).T) / np.sum(
                np.multiply(s_1_probabilities, transition_matrix.T).T)

            # create a prediction using MAP
            weighted_mu = np.sum(np.multiply(s_2_probabilities, state_features['MU']))
            predicted_values.append(weighted_mu)

            # state 2 probabilities now become state 1 probabilities (this becomes increasingly unstable further OOB
            s_1_probabilities = s_2_probabilities

        predicted_values = np.divide(predicted_values, self.scale_value)

        return predicted_values


class HiddenMarkovModel:

    def __init__(self, timeseries, n_states, scale_value):
        """
        Args:
            timeseries: list/series, timeseries to fit HMM upon
            n_states: int, number of states within HMM
            scale_value: float, value to scale the timeseries by, strictly for pdf/cdf functions
        """
        self.timeseries = timeseries * scale_value
        self.scale_value = scale_value
        self.n_states = n_states

    def fit_hmm(self):
        """
        Wrapper for fitting a HMM to a timeseries of returns.
        This function is effectively defining the hidden state spaces and producing fitted results
        given the timeseries, hidden states, and transition matrix... all calibrated within
        the baum-welch algorithm.
        """
        transition_matrix, state_features, observation_probabilities = self.run_baum_welch_algorithm()
        state_list, forward_probabilities = self.run_viterbi_algorithm(state_features, transition_matrix)
        fitted_values = self.fit_values(forward_probabilities, transition_matrix, state_features)

        return HiddenMarkovModelResults(self.n_states, transition_matrix, state_features, fitted_values,
                                        forward_probabilities, self.scale_value)

    def fit_values(self, observation_probabilities, transition_matrix, state_features):
        """
        This function predicts the value given the state features and observation probabilities.

        Args:
            - observation_probabilities: array, the fwd-bkwd derived probabilities of observing i from state j
            - transition_matrix: array, the probability of transitioning from state i to state j
            - state_features: dataframe, the state parameters of the hidden states
        Returns:
            - fitted_values: list, the fitted predictions
        """
        fitted_values = [0]

        for observation in observation_probabilities:
            future_state_probs = sum(np.multiply(observation, transition_matrix.T).T) / np.sum(
                np.multiply(observation, transition_matrix.T).T)

            weighted_mu = np.sum(np.multiply(future_state_probs, state_features['MU']))

            fitted_values.append(weighted_mu)

        fitted_values = np.divide(fitted_values, self.scale_value)

        # drop the last value, as that is OOB
        fitted_values = fitted_values[:-1]

        return fitted_values

    def run_viterbi_algorithm(self, state_features, transition_matrix):
        """
        This function runs the viterbi algorithm. It considers the most likely forward pass
        of observations running through hidden states with defined features.

        Args:
            - state_features: dataframe, housing the features of our hidden states
            - transition_matrix: array, housing the probability of transition from state i --> state j
        Returns:
            - state_list: list, the MAP hidden state as defined by the viterbi algorithm
            - forward_probabilities: array, the probabilities of each state at step i
        """

        # initialized variables
        observation_probabilities = self.create_observation_probabilities(state_features)

        alpha = observation_probabilities[0]
        forward_probabilities = [alpha / sum(alpha)]
        forward_trellis = [np.array([alpha] * self.n_states) / np.sum(np.array([alpha] * self.n_states))]

        # Given the probability of the initial observation in the initial state, get the probabiltiy of a transition p(t|p(o|s))
        for i in range(1, len(observation_probabilities)):
            # the probabibility of obervation k coming from states i:j
            observation_probability = observation_probabilities[i]

            # the probability of moving from state 1ij to state 2ij (given the starting probability alpha)
            state_to_state_probability = np.multiply(alpha, transition_matrix.T).T

            # The probability of observation i coming from state i,j
            forward_probability = np.multiply(observation_probability, state_to_state_probability)
            forward_probability = forward_probability / np.sum(forward_probability)

            # Re-evaluate alpha (probability of being in state i)
            alpha = sum(forward_probability) / np.sum(forward_probability)
            forward_trellis.append(forward_probability)
            forward_probabilities.append(alpha)

        # create empty list to store the states
        state_list = []
        forward_trellis.reverse()

        prev_state = np.where(forward_trellis[0] == np.max(forward_trellis[0]))[1][0]

        # for each step, evaluate the MAP of the state coming from one of the subsequent states
        for member in forward_trellis:
            state = np.where(member == np.max(member[:, prev_state]))[0][0]
            state_list.append(state)
            prev_state = state
        state_list.reverse()

        return state_list, forward_probabilities

    def run_baum_welch_algorithm(self):
        """
        Run a forward backward algorithm on a suspected HMM (Hidden Markov Model). This
        is the step where the parameters for a HMM are fit to the data

        Args:

        Returns:
            - transition_matrix: array, probability of transitioning from state i to state j
            - state_features: dataframe, defining features of a state space
            - probabilities: array, the relative probabilities of observation i coming from state j
        """

        # Start by defining the mean and variance of each state
        state_features = self.create_initial_state_features()

        # create the initial transition matrix
        transition_matrix = self.create_transition_matrix()

        for i in range(10):
            # create the observation probabilities given the initial features
            observation_probabilities = self.create_observation_probabilities(state_features)

            # run forward and backward pass through
            forward_probabilities, forward_trellis = self.run_forward_pass(transition_matrix, observation_probabilities)
            backward_probabilities, backward_trellis = self.run_backward_pass(transition_matrix,
                                                                              observation_probabilities)
            backward_trellis.reverse()
            backward_probabilities.reverse()

            # update lambda parameter (probability of state i, time j)
            numerator = np.multiply(np.array(forward_probabilities), np.array(backward_probabilities))

            denominator = sum(np.multiply(np.array(forward_probabilities), np.array(backward_probabilities)).T)
            _lambda = []
            for j in range(len(numerator)):
                _lambda.append((numerator[j, :].T / denominator[j]).T)

            # update epsilon parameter (probability of moving for state i to state j)
            numerator = np.multiply(forward_trellis[1:], backward_trellis[:-1])
            epsilon = []
            for g in range(len(numerator)):
                denominator = np.sum(numerator[g, :, :])
                epsilon.append((numerator[g, :, :].T / denominator).T)

            # Update the transition matrix and observation probabilities for the next iteration
            transition_matrix = ((sum(epsilon) / sum(_lambda))).T / sum((sum(epsilon) / sum(_lambda)))

            # Update the state space parameters
            observation_probabilities = _lambda
            for state in range(self.n_states):
                state_weight = 0
                state_var = 0
                state_sum = 0
                for obs in range(len(self.timeseries)):
                    state_weight += _lambda[obs][state]
                    state_sum += self.timeseries[obs] * _lambda[obs][state]
                    state_var += _lambda[obs][state] * np.sqrt(
                        (self.timeseries[obs] - state_features.loc[state_features['STATE'] == state, 'MU']) ** 2)

                state_features.loc[state_features['STATE'] == state, 'MU'] = state_sum / state_weight
                state_features.loc[state_features['STATE'] == state, 'SIGMA'] = state_var / state_weight

        print('Fitted transition matrix: ')
        print(transition_matrix)
        print('Fitted state features: ')
        print(state_features)

        # multiply the probabilities to get the overall probability. Convert to state using MAP
        observation_probabilities = _lambda

        return transition_matrix, state_features, observation_probabilities

    def create_transition_matrix(self):
        """
        This function creates the initial transition matrix for our HMM.
        We initialize the transition probabilities to equal, however these
        transition probabilities will be adapted as we perform forward and
        backward passes in the broader fwd-bkwd algorithm.

        Args:
            - None
        Returns:
            - transition_matrix: array, transition matrix
        """

        # For each state, create a transition probability for state_i --> state_j
        # We initialize the transition probabilites as decreasing to more distant states
        transition_list = []
        for state in range(self.n_states):
            init_probs = [1] * self.n_states
            init_mult = [(1 / ((abs(x - state) + 1) * 1.5)) * init_probs[x] for x in range(len(init_probs))]
            state_transition_prob = np.divide(init_mult, sum(init_mult))
            transition_list.append(state_transition_prob)

        transition_matrix = np.array(transition_list)
        print('Initial transition matrix created for {} states: '.format(self.n_states))
        print(transition_matrix)
        return transition_matrix

    def run_forward_pass(self, transition_matrix, observation_probabilities):
        """
        The forward pass of a forward backward algorithm. Calculating the forward probabilities

        Args:
            - transition_matrix: array, probability of transitioning from state i to state j
            - observation_probabilities: array, probability of observation i coming from state j
        Returns:
            - forward_results: array, calculated forward probabilities
            - forward_trellis: trellis of stored results from forward pass
        """

        # initialize the variables
        alpha = observation_probabilities[0]
        forward_results = [alpha]
        forward_trellis = [np.array([alpha] * self.n_states) / np.sum(np.array([alpha] * self.n_states))]

        # Given the probability of the initial observation in the initial state, get the probability of a transition p(t|p(o|s))
        for i in range(1, len(observation_probabilities)):
            # the probability of observation k coming from states i:j
            observation_probability = observation_probabilities[i]

            # the probability of moving from state 1ij to state 2ij (given the starting probability alpha)
            state_to_state_probability = np.multiply(alpha, transition_matrix.T).T

            # The probability of observation i coming from state i:j
            forward_probability = np.multiply(observation_probability, state_to_state_probability)
            forward_probability = forward_probability / np.sum(forward_probability)

            # Re-evaluate alpha (probability of being in state i at step)
            alpha = sum(forward_probability)
            forward_trellis.append(forward_probability)
            forward_results.append(alpha)

        return forward_results, forward_trellis

    def run_backward_pass(self, transition_matrix, observation_probabilities):
        """
        The backward pass of a forward backward algorithm. Calculating the backward probabilities

        Args:
            - transition_matrix = array, probability of transitioning from state i to state j
            - observation_probabilities: array, probability of observation i coming from state j
        Returns:
            - backward_results: array, calculated backward probabilities
            - backward_trellis: trellis of stored results from backward pass
        """
        # initialize variables
        beta = [1] * self.n_states
        backward_results = [beta]
        backward_trellis = [np.array([beta] * self.n_states)]

        # Given the probability of the initial observation in the initial state, get the probability of a transition p(t|p(o|s))
        for i in range(2, len(observation_probabilities) + 1):
            # the probability of observation k coming from states i:j
            observation_probability = observation_probabilities[-i]

            # the probability of moving from state 1ij to state 2ij (given the starting probability alpha)
            state_to_state_probability = np.multiply(beta, transition_matrix)

            # The probability of observation i coming from state i,j
            backward_probability = np.multiply(observation_probability, state_to_state_probability.T).T
            backward_probability = backward_probability / np.sum(backward_probability)

            # Re-evaluate beta (probability of being in state i at step)
            beta = sum(backward_probability.T)
            backward_trellis.append(backward_probability)
            backward_results.append(beta)

        return backward_results, backward_trellis

    def create_initial_state_features(self):
        """
        This function aims to define the features of the initial state-spaces (hidden).
        We do this by partitioning the data into quantile sets and creating mu and sigma parameters
        that define the quantile.

        Args:

        Returns:
            - initial_state_features: dataframe, containing the parameter set of the state-spaces
        """

        # the percentile blocks will be used to break our range of observations into "buckets"
        percentile_blocks = 100 / self.n_states

        # Using the percentile blocks, we define the features of each given state
        # these definitions will be refined using the EM baum-welch algo
        state_list = []
        mu_list = []
        sigma_list = []
        for i in range(self.n_states):
            upper_bound = np.percentile(self.timeseries, (i + 1) * percentile_blocks)
            lower_bound = np.percentile(self.timeseries, i * percentile_blocks)

            price_subset = self.timeseries[(self.timeseries >= lower_bound) & (self.timeseries < upper_bound)]
            mu = price_subset.mean()
            sigma = price_subset.std()

            state_list.append(i)
            mu_list.append(mu)
            sigma_list.append(sigma)

        initial_state_features = pd.DataFrame(
            {'STATE': state_list, 'MU': mu_list, 'SIGMA': sum(sigma_list) / len(sigma_list)})

        return initial_state_features

    def create_observation_probabilities(self, state_features):
        """
        Evaluate the observation probabilities for each state, given the
        state features.

        Args:
            - state_features: dataframe/array, containing the parameter set of a given state-space
        Returns:
            - observation_probabilities: dataframe/array, containing the probability of our observations from a given state0
        """
        observation_state_container = []

        for _, row in state_features.iterrows():
            state_probabilities = norm.pdf(self.timeseries, loc=row['MU'], scale=row['SIGMA'])
            observation_state_container.append(state_probabilities)

        observation_state_probabilities = np.array(observation_state_container).T

        return observation_state_probabilities
