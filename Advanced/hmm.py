# Hidden Markov Model - Forward Algorithm
import numpy as np
from hmmlearn import hmm


def forward_algorithm(obs, hidden_states,
                      transition_matrix, emission_matrix,
                      init=np.array([1/3, 1/3, 1/3])):
    T = len(obs)  # number of days
    N = len(hidden_states)  # number of hidden states
    # Forward probability matrix: rows=time, columns=states
    alpha = np.zeros((T, N))
    # Day 1 initialization
    O0 = obs_map[observations[0]]  # index of first observation
    alpha[0, :] = init * emission_matrix[:, O0]
    print("alpha[0]:", alpha[0])
    # Compute alpha for days 2 to T
    for t in range(1, T):
        Ot = obs_map[observations[t]]  # current observation index
        for j in range(N):
            # sum over previous states
            alpha[t, j] = (np.sum(alpha[t - 1, :] * transition_matrix[:, j])
                           * emission_matrix[j, Ot])
        print(f"alpha[{t}]:", alpha[t])

    # Sum of alpha at final time step gives probability of observing the sequence
    P_O = np.sum(alpha[T - 1, :])
    print("\nProbability of the observation sequence:", P_O)


# Observations and states
observations = ["Walk", "Shop", "Clean", "Walk", "Walk", "Shop", "Clean", "Walk", "Clean", "Shop"]
hidden_states = ["Sad", "Happy", "Angry"]

# Mapping for convenience
obs_map = {"Walk": 0, "Clean": 1, "Shop": 2}  # column indices in B
obs_seq = np.array([obs_map[o] for o in observations]).reshape(-1, 1)  # shape (n_samples, 1)

state_map = {"Sad": 0, "Happy": 1, "Angry": 2}  # row indices

# Transition matrix A (state -> state)
# Row order Sad, Happy, Angry; Column order: Walk, Clean, Shop.
A = np.array([[0.4, 0.4, 0.2],
              [0.1, 0.8, 0.1],
              [0.3, 0.2, 0.5]])

# Emission matrix B (state -> observation)
B = np.array([[0.1, 0.2, 0.7],
              [0.6, 0.3, 0.1],
              [0.2, 0.6, 0.2]])

# Initial probabilities (uniform)
pi = np.array([1/3, 1/3, 1/3])
# Call the function
forward_algorithm(observations, hidden_states, A, B, pi)

# CategoricalHMM for discrete observations
# Unsupervised Learning (the hidden states are never known)
print("\nHidden Markov Model:")
model = hmm.CategoricalHMM(n_components=3, n_iter=100, tol=0.01, verbose=True)
# Since we don't know initial probabilities, transitions, or emissions, let it learn
model.fit(obs_seq)
# Print learned parameters
print("Learned start probabilities (pi):\n", model.startprob_)
print("\nLearned transition matrix (A):\n", model.transmat_)
print("\nLearned emission matrix (B):\n", model.emissionprob_)

# Decode hidden states for the sequence
log_prob, hidden_states = model.decode(obs_seq, algorithm="viterbi")
print("\nMost likely hidden states sequence:", hidden_states)
