import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import time
START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'


def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
#    print('data[10]')
#    print(data[10])
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])

    print(pos)



class Baseline(object):
    '''
    The baseline model.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        
        train_pos = training_set[:, 0]
        train_words = training_set[:, 1]
        
        all_pos_train = np.concatenate(train_pos)
        all_words_train = np.concatenate(train_words)
        
        all_pos_train2i = np.vectorize(self.pos2i.get)(all_pos_train).reshape((all_pos_train.shape[0],1))
        all_words_train2i = np.vectorize(self.word2i.get)(all_words_train).reshape((all_words_train.shape[0],1))
        #calculates P(y)        
        pos_unique, pos_counts = np.unique(all_pos_train2i, return_counts=True)
        pos_sum_counts = np.sum(pos_counts)
        self.pos_prob = np.zeros((self.pos_size,))
        self.pos_prob[pos_unique] = pos_counts / pos_sum_counts
        self.pos_prob = self.pos_prob.reshape((self.pos_size,1))
        
        #calculates p(y intersection x)
        pos_words_pairs = np.concatenate((all_pos_train2i ,all_words_train2i), axis=1)
        pos_words_unique, pos_words_counts = np.unique(pos_words_pairs, return_counts=True, axis = 0)
        pos_words_sum_counts = np.sum(pos_words_counts)
        self.pos_words_prob = np.zeros((self.pos_size, self.words_size))
        self.pos_words_prob[pos_words_unique[:,0], pos_words_unique[:,1]] = pos_words_counts / pos_words_sum_counts
        
        #calculates the emission matrix
        tiled_pos_prob =  np.tile(self.pos_prob, self.words_size)
        self.emission = np.divide(self.pos_words_prob , tiled_pos_prob,
                                           out=np.zeros_like(self.pos_words_prob), where=tiled_pos_prob!=0)

		
    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        max_words_prob_indices = np.argmax(self.pos_words_prob, axis=0)
        np_pos_tags = np.array(self.pos_tags)
        pos_sequence = sentences.copy()
        i = 0
        for sentence in sentences:
            sentence2i = np.vectorize(self.word2i.get)(sentence)
            pos2i_max = max_words_prob_indices[sentence2i]
            pos_sequence[i] = np_pos_tags[pos2i_max]
            i += 1
        return pos_sequence
		
def baseline_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. You may implement
            the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    baseline_model =  Baseline(model.pos_tags, model.words, training_set)
    return baseline_model.pos_prob,  baseline_model.emission
		
class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        self.emission = baseline_mle(training_set, self)[1]
        train_pos_sentences = training_set[:, 0]
        self.transition_matrix = np.zeros((self.pos_size, self.pos_size))
        for pos_sentence in train_pos_sentences:
            pos_sentence2i = np.vectorize(self.pos2i.get)(pos_sentence)
            for (i,j) in zip(pos_sentence2i,pos_sentence2i[1:]):
                self.transition_matrix[i][j] += 1
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = np.divide(self.transition_matrix , row_sums[:, np.newaxis],
                                           out=np.zeros_like(self.transition_matrix ), where=row_sums[:, np.newaxis]!=0)
    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''
        sequences = []
        for i in range(n):
            current_pos = START_STATE
            sequence = [START_WORD]
            while current_pos != END_STATE:         
                current_pos2i = self.pos2i[current_pos]
                pos_probs = self.transition_matrix[current_pos2i]
                current_pos_index = np.random.choice(self.pos_size, p = pos_probs)
                current_pos = self.pos_tags[current_pos_index]
                current_pos2i = self.pos2i[current_pos]
                word_probs = self.emission[current_pos2i]
                word_index = np.random.choice(self.words_size, p = word_probs)
                word = self.words[word_index]
                sequence.append(word)
            sequences.append(sequence)
        return sequences


    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        log_transition = np.log(self.transition_matrix + np.finfo(np.float64).tiny)
        log_emission = np.log(self.emission + np.finfo(np.float64).tiny)
        sequence_pos = sentences.copy()
        k = 0
        np_pos_tags = np.array(self.pos_tags)
        for sentence in sentences:
            pi = np.zeros((len(sentence), self.pos_size))
            for t in range(len(sentence)):
                for i in range(self.pos_size):
                    if t == 0:
                        pi[t, i] = log_transition[self.pos2i[START_STATE], i] + log_emission[i, self.word2i[sentence[t]]]
                    else:
                        pi[t,i] = np.max(pi[t-1] + log_transition[:, i] + log_emission[i ,self.word2i[sentence[t]]])
            max_incdices = np.argmax(pi, axis=1)
            sentence_pos_tags = np_pos_tags[max_incdices]
            sequence_pos[k] = sentence_pos_tags
            k += 1
        return sequence_pos
def hmm_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    hmm_model =  HMM(model.pos_tags, model.words, training_set)
    return hmm_model.transition_matrix,  hmm_model.emission 

def phi(pos_tag1, pos_tag2, word, model):
    """
    the feature mapping function, which accepts two PoS tags
    and a word, and returns a list of indices that have a "1" in
    the binary feature vector.
    :param pos_tag1: y(t-1)
    :param pos_tag2: y(t)
    :param word: x(t)
    :param model: an MEMM  model
    """
    pos_tag_i = model.pos_size * model.words_size + model.pos2i[pos_tag1] * model.pos_size + model.pos2i[pos_tag2]
    word_i = model.pos2i[pos_tag2] * model.words_size + model.word2i[word]
    return [word_i, pos_tag_i]

def phi_ing(pos_tag1, pos_tag2, word, model):
    """
    the feature mapping function, which accepts two PoS tags
    and a word, and returns a list of indices that have a "1" in
    the binary feature vector.
    adds features for word that end with ing
    :param pos_tag1: y(t-1)
    :param pos_tag2: y(t)
    :param word: x(t)
    :param model: an MEMM  model
    """
    indices = phi(pos_tag1, pos_tag2, word, model)
    indices_from = model.pos_size * model.words_size + model.pos_size ** 2
    if word[-3:] == "ing":
        word_i = model.pos2i[pos_tag2] * model.words_size + model.word2i[word]
        ing_indices =  indices_from + word_i
        indices.append(ing_indices)
    return indices

def phi_ed(pos_tag1, pos_tag2, word, model):
    """
    the feature mapping function, which accepts two PoS tags
    and a word, and returns a list of indices that have a "1" in
    the binary feature vector.
    adds features for word that end with ed
    :param pos_tag1: y(t-1)
    :param pos_tag2: y(t)
    :param word: x(t)
    :param model: an MEMM  model
    """
    indices = phi(pos_tag1, pos_tag2, word, model)
    indices_from = model.pos_size * model.words_size + model.pos_size ** 2
    if word[-2:] == "ed":
        word_i = model.pos2i[pos_tag2] * model.words_size + model.word2i[word]
        ed_indices =  indices_from + word_i
        indices.append(ed_indices)
    return indices

def phi_ly(pos_tag1, pos_tag2, word, model):
    """
    the feature mapping function, which accepts two PoS tags
    and a word, and returns a list of indices that have a "1" in
    the binary feature vector.
    adds features for word that end with ly
    :param pos_tag1: y(t-1)
    :param pos_tag2: y(t)
    :param word: x(t)
    :param model: an MEMM  model
    """
    indices = phi(pos_tag1, pos_tag2, word, model)
    indices_from = model.pos_size * model.words_size + model.pos_size ** 2
    if word[-2:] == "ly":
        word_i = model.pos2i[pos_tag2] * model.words_size + model.word2i[word]
        ly_indices =  indices_from + word_i
        indices.append(ly_indices)
    return indices
def phi_upper_case(pos_tag1, pos_tag2, word, model):
    """
    the feature mapping function, which accepts two PoS tags
    and a word, and returns a list of indices that have a "1" in
    the binary feature vector. 
    adds features for word that begin with upper case letters
    :param pos_tag1: y(t-1)
    :param pos_tag2: y(t)
    :param word: x(t)
    :param model: an MEMM  model
    """
    indices = phi(pos_tag1, pos_tag2, word, model)
    indices_from = model.pos_size * model.words_size + model.pos_size ** 2
    if word[0].isupper():
        word_i = model.pos2i[pos_tag2] * model.words_size + model.word2i[word]
        upper_indices =  indices_from + word_i
        indices.append(upper_indices)
    return indices

def get_sum_w(w, word, pos_tag1, pos_tag2, model):
    """
    function that sums w on the right indices
    :param w: a weights vector.
    :param word: a word
    :param pos_tag1: y(t-1)
    :param pos_tag2: y(t)
    :param model: an MEMM  model
    """
    
    indices = model.phi(pos_tag1, pos_tag2, word, model)
    return w[indices].sum()

def get_z(word, pos_tag, pos_tags, model, w):
    z_sum = 0
    for tag in pos_tags:
         z_sum += np.exp(get_sum_w(w, word, pos_tag, tag, model))
    return z_sum
        
class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, training_set, phi):

        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos:i for (i,pos) in enumerate(pos_tags)}
        self.word2i = {word:i for (i,word) in enumerate(words)}
        self.phi = phi


    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''
        sequence_pos = sentences.copy()
        k = 0
        np_pos_tags = np.array(self.pos_tags)

        for sentence in sentences:
            pi = np.zeros((len(sentence), self.pos_size))
            for t in range(1,len(sentence)):
                if t != 1:
                    z_values = [np.log(get_z(sentence[t], np_pos_tags[j], np_pos_tags, self, w)) for j in range(self.pos_size)]
                
                for i in range(self.pos_size):
                    if t == 1:
                        pi[t, i] = get_sum_w(w, sentence[t], START_STATE, np_pos_tags[i], self) - \
                        np.log(get_z(sentence[t], START_STATE, np_pos_tags, self, w)) 
                    else:
                        pi[t,i] = np.max([pi[t-1, j] + get_sum_w(w, sentence[t], np_pos_tags[j], np_pos_tags[i], self)  - 
                                   z_values[j] for j in range(self.pos_size)])
            max_incdices = np.argmax(pi, axis=1)
            sentence_pos_tags = np_pos_tags[max_incdices]
            sequence_pos[k] = sentence_pos_tags
            k += 1
        return sequence_pos


def perceptron(training_set, initial_model, w0, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    N = training_set.shape[0]
#    memm_model = MEMM(initial_model.pos_tags, initial_model.words, training_set, initial_model.phi)
    w = w0
    sentences = training_set[:,1]
    pos_tags = training_set[:,0]
    for epoch in range(epochs):
        for i in range(N):
            y_hat = initial_model.viterbi([sentences[i]], w)[0]
            for t in range(1, len(sentences[i])):
                pos_tags_indices = initial_model.phi(pos_tags[i][t-1], pos_tags[i][t], sentences[i][t], initial_model)
                y_hat_indices = initial_model.phi(y_hat[t-1], y_hat[t], sentences[i][t], initial_model)
                w[pos_tags_indices]  +=  eta
                w[y_hat_indices]  -=  eta
    return w
def data_split(data):
    """ 
    splits the data into training set and test set
    :param data_path: the path of the PoS_data file
    :return: tuple train set , test set as numpy arrays
    """
    train_set, test_set = train_test_split(data)
    return np.array(train_set), np.array(test_set)

def add_start_end_states(data):
    for i in range(len(data)):
        sentences = [START_WORD] + data[i][1] + [END_WORD]
        pos =  [START_STATE] + data[i][0] + [END_STATE]
        data[i] = (pos, sentences)
    return data

def check_success(result_pos, pos_data):
    avg_success = 0 
    for i in range(result_pos.shape[0]):
        np_pos_data = np.array(pos_data[i])
        np_result_pos = np.array(result_pos[i])
        bool_array = (np_pos_data == np_result_pos)
        sentence_success = np.sum(bool_array) 
        avg_success += sentence_success
    avg_success /= np.concatenate(result_pos).shape[0]
    return avg_success

def handle_rare_words(data, words, n):
    sentences = data[:, 1]
    all_words_data = np.concatenate(sentences)
    words_unique, words_counts = np.unique(all_words_data, return_counts=True)
    words_counter = dict(zip(words_unique, words_counts))
    rare_word = "*rare*"
    for word in words:
        if word in words_counter:
            if words_counter[word] <= n:
                words.remove(word)
        else:
            words.remove(word)
    words.append(rare_word)
    for i in range(len(data)):
        for w in range(len(data[i][1])):
            if words_counter[data[i][1][w]] <= n:
                data[i][1][w] = rare_word
    return data, words
    
if __name__ == '__main__':

#    data_example()
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    data = np.array(data)
    data, words = handle_rare_words(data, words, 2)
    words =  [START_WORD] + words + [END_WORD]
    pos = [START_STATE] + pos + [END_STATE]
    
    data = add_start_end_states(data)
    len_data = len(data)
    
    test_set = data[int(len_data * 0.9):]
    test_sentences = test_set[:,1]
    test_pos_data = test_set[:,0]
    
    #running baseline
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        start = time.time()
        baseline = Baseline(pos, words, train_set)
        end = time.time()
        print("finished training baseline with time", end - start)
        start = time.time()
        baseline_pos = baseline.MAP(test_sentences)
        end = time.time()
        print("finished labeling baseline with time", end - start)
        print("success rate on baseline with train percentage: ", percentage)
        print(check_success(baseline_pos, test_pos_data))
        pos_prob , emission = baseline_mle(train_set, baseline)
    

    #running hmm
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        start = time.time()
        hmm = HMM(pos, words, train_set)
        end = time.time()
        print("finished training hmm with time", end - start)
        start = time.time()
        hmm_pos = hmm.viterbi(test_sentences)
        end = time.time()
        print("finished labeling (viterbi) hmm with time", end - start)
        print("success rate on hmm with train percentage: ", percentage)
        print(check_success(hmm_pos, test_pos_data))
    
    ## showing sampling exmple on hmm with n =5
    print("showing sampling exmple on hmm with n =5")
    print(hmm.sample(5))
    
    #running memm with basic phi (emission and transition)
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        memm = MEMM(pos, words, train_set, phi)
        w0 = np.zeros(memm.words_size * memm.pos_size + memm.pos_size ** 2 )
        start = time.time()
        w = perceptron(train_set, memm, w0, eta=0.1, epochs=1)
        end = time.time()
        print("finished perceptron MEMM with time", end - start)
        start = time.time()
        memm_pos = memm.viterbi(test_sentences, w)
        end = time.time()
        print("finished viterbi MEMM with time", end - start)
        print("success rate on memm basic phi with train percentage: ", percentage)
        print(check_success(memm_pos, test_pos_data))
    
    #running memm with phi ing +  (emission and transition)
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        memm = MEMM(pos, words, train_set, phi_ing)
        w0 = np.zeros(memm.words_size * memm.pos_size * 2 + memm.pos_size ** 2 )
        start = time.time()
        w = perceptron(train_set, memm, w0, eta=0.1, epochs=1)
        end = time.time()
        print("finished perceptron MEMM with time", end - start)
        start = time.time()
        memm_pos = memm.viterbi(test_sentences, w)
        end = time.time()
        print("finished viterbi MEMM with time", end - start)
        print("success rate on memm ing phi with train percentage: ", percentage)
        print(check_success(memm_pos, test_pos_data))
    
    #running memm with phi ed +  (emission and transition)
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        memm = MEMM(pos, words, train_set, phi_ed)
        w0 = np.zeros(memm.words_size * memm.pos_size * 2 + memm.pos_size ** 2 )
        start = time.time()
        w = perceptron(train_set, memm, w0, eta=0.1, epochs=1)
        end = time.time()
        print("finished perceptron MEMM with time", end - start)
        start = time.time()
        memm_pos = memm.viterbi(test_sentences, w)
        end = time.time()
        print("finished viterbi MEMM with time", end - start)
        print("success rate on memm ed phi with train percentage: ", percentage)
        print(check_success(memm_pos, test_pos_data))
    
    #running memm with phi ly +  (emission and transition)
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        memm = MEMM(pos, words, train_set, phi_ly)
        w0 = np.zeros(memm.words_size * memm.pos_size * 2 + memm.pos_size ** 2 )
        start = time.time()
        w = perceptron(train_set, memm, w0, eta=0.1, epochs=1)
        end = time.time()
        print("finished perceptron MEMM with time", end - start)
        start = time.time()
        memm_pos = memm.viterbi(test_sentences, w)
        end = time.time()
        print("finished viterbi MEMM with time", end - start)
        print("success rate on memm ly phi with train percentage: ", percentage)
        print(check_success(memm_pos, test_pos_data))
    
    #running memm with phi upper case +  (emission and transition)
    for percentage in [0.1, 0.25, 0.9]:
        train_set = data[:int(len_data * percentage)]
        memm = MEMM(pos, words, train_set, phi_upper_case)
        w0 = np.zeros(memm.words_size * memm.pos_size * 2 + memm.pos_size ** 2 )
        start = time.time()
        w = perceptron(train_set, memm, w0, eta=0.1, epochs=1)
        end = time.time()
        print("finished perceptron MEMM with time", end - start)
        start = time.time()
        memm_pos = memm.viterbi(test_sentences, w)
        end = time.time()
        print("finished viterbi MEMM with time", end - start)
        print("success rate on memm upper case phi with train percentage: ", percentage)
        print(check_success(memm_pos, test_pos_data))
