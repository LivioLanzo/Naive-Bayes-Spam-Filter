import re
from collections import defaultdict
import math
import random


class SpamFilter(object):

    def __init__(self):
        self._number_of_spam_messages = 0
        self._number_of_non_spam_messages = 0
        self._words_count = defaultdict(lambda: [0, 0])
        self._words_probabilities = defaultdict(lambda: [0.0, 0.0])

    @staticmethod
    def _tokenize_words(message):
        '''
            DESCRIPTION
            --------------
            returns a list of unique words found within a message
            words are considered to be containing letters, numbers, single apostrophes and dashes

            PARSMETERS
            -------------
            message: a string we wish to tokenize
        '''
        # extract single words
        words = re.findall(r"[a-z0-9']+", message, re.IGNORECASE)

        # make them lower case and return unique list
        return set(word.lower() for word in words)

    def _add_messages(self, messages):
        '''
            DESCRIPTION
            ----------
            keeps a tally of the words appearing in each message and 
            of the overall spam and non-spam messages.

            self._words_count if a dict where each element is a
            two-elements lists. The first element of the list keeps 
            a tally of the spam messages in which the key word appeared, the
            second element of the list keeps a tally of the non spam
            messages in which the key word appeared

            PARAMETERS
            ---------
            messages: a list of tuples, where the first element is
                      either True (spam message) or False (non spam message)
                      and the second element is the message itself

            RETURNS
            ---------
            None
        '''
        for is_spam, message in messages:
            # add counts based on the category
            if is_spam:
                self._number_of_spam_messages += 1
            else:
                self._number_of_non_spam_messages += 1

            # tokenize words
            words = self._tokenize_words(message)

            for word in words:
                if is_spam:
                    self._words_count[word][0] += 1
                else:
                    self._words_count[word][1] += 1

        self._recalculate_probabilities()

    def _recalculate_probabilities(self):
        '''
            DESCRIPTION
            -----------
            recalculates the conditional probability of each word
            given spam and given not spam

            to avoid the issue of when a probability is zero (which would result into a zero product)
            we will use additive smoothing: https://en.wikipedia.org/wiki/Additive_smoothing
        '''
        for word, (spam_count, non_spam_count) in self._words_count.items():
            # the probability of a spam message containing word X
            self._words_probabilities[word][0] = (
                0.5 + spam_count) / (1 + self._number_of_spam_messages)
            # the probability of a non spam message containing word X
            self._words_probabilities[word][1] = (
                0.5 + non_spam_count) / (1 + self._number_of_non_spam_messages)

    def train(self, messages):

        self._add_messages(messages)

    def probability_spam(self, message):

        '''
            DESCRIPTION
            -----------
            returns the probability of a message being spam
            adds the natural logarithm of each probability 

            PARAMETERS
            ----------
            a message string

            RETURNS
            ---------
            the probability of message being spam
        '''

        spam_probability = math.log((0.5 + self._number_of_spam_messages) / (
            1 + self._number_of_spam_messages + self._number_of_non_spam_messages))
        non_spam_probability = math.log((0.5 + self._number_of_non_spam_messages) / (
            1 + self._number_of_spam_messages + self._number_of_non_spam_messages))


        # extract unique words from message
        words_in_message = self._tokenize_words(message)

        for word, (word_spam_prob, word_non_spam_prob) in self._words_probabilities.items():
            # if word is found in words_in_message then add the log in base e of the probabilities
            # of word appearing in a spam message and word appearinging in a non spam message
            if word in words_in_message:
                spam_probability += math.log(word_spam_prob)
                non_spam_probability += math.log(word_non_spam_prob)
            # if word is found in words_in_message then add the log in base e of the probabilities
            # of word not appearing (1-probability of appearing in a spam message) in a spam message
            # and word not appearing in a non spam message (1-probability of appearing in a non spam message)
            else:
                spam_probability += math.log(1.0 - word_spam_prob)
                non_spam_probability += math.log(1.0 - word_non_spam_prob)
        
        return math.exp(spam_probability) / (math.exp(spam_probability) + math.exp(non_spam_probability))



def split_train_test(data: list, test_size: float=0.4):

    '''
        DESCRIPTION
        -----------
        divides data into a training set and a testing set
        test_size is used to define the % of data which should go into
        the testing set

        PARAMETERS
        ----------
        a list

        RETURNS
        ----------
        a tuple where the first element is the training set
        ans the second element is the testing set
    '''
    # % must be between 0.1 and 0.99
    if not 0.99 > test_size > 0.1:
        raise ValueError('test_size must be between 0.1 and 0.99')
    
    test_size_int = int(test_size * 100)

    train_data = []
    test_data = []

    for row in data:
        if random.randint(1, 100) <= test_size_int:
            test_data.append(row)
        else:
            train_data.append(row)
    
    return train_data, test_data




if __name__ == '__main__':

    messages = []

    # read the txt file
    with open('SMSSpamCollection.txt', mode='r', encoding='UTF-8') as spam_file:

        for line in spam_file:
            category, message = line.split('\t', maxsplit=1)
            messages.append((category=='spam', message))


    # split data into training and testing

    train, test = split_train_test(messages, 0.2)

    # train the model
    s  = SpamFilter()
    s.train(train)

    # test the model
    stats = {'True Positive': 0, 'False Positive': 0, 'True Negative': 0, 'False Negative': 0}

    for is_spam, message in test:
        spam_prob = s.probability_spam(message) > 0.5
        if spam_prob:
            if is_spam:
                stats['True Positive'] += 1
            else:
                stats['False Positive'] +=1
        else:
            if is_spam:
                stats['False Negative'] += 1
            else:
                stats['True Negative'] += 1

    print(stats)
    print('Precision: {0:.2%}'.format(stats['True Positive'] / (stats['True Positive'] + stats['False Positive'])))
    print('\n')