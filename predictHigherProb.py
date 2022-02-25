import numpy as np

# Using just the (unscaled) complexity of series x and series y, can we predict whether series x will have higher or lower probability than series y? In theory, if x is simpler than y, then we should predict that x has higher probability, and vice versa. We test this now

def PredictWhichIsHigherProb(P,K):
    assert(len(P)==len(K))
    assert(np.abs(sum(P)-1)<0.001)

    success_rate = []
    for samps in range(10000):# do samples to test our predictions
        # pick a random series x and series y, according to their probabilities, and record their probabilities
        indx = np.random.choice(np.arange(len(K)),p=P)
        indy = np.random.choice(np.arange(len(K)),p=P)
        if K[indx] < K[indy]:
            #predict x is more likely that y (or equal)
            success_rate.append(1*(P[indx]>=P[indy]))
        elif K[indy] < K[indx]:
            success_rate.append(1*(P[indy]>=P[indx]))
        elif K[indy] == K[indx]:
            #if the complexities are the same, flip a coin to predict which has higher probability
            success_rate.append(1*(np.random.rand()>0.5))

    return np.sum(success_rate)/len(success_rate)

# list of probabilities of the binary patterns
list_probs = [0.1, 0.2, 0.3, 0.4-0.01,0.01]

# list of complexities of the binary patterns
list_K = [5,4,3,2,6]

prediction_success_rate = PredictWhichIsHigherProb(list_probs,list_K)

print('The prediction success rate is',prediction_success_rate)