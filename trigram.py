import string
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.structure import TanhLayer, LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError


CORPUS_LENGTH = 300

trigram = list()
word_list = list()
temp_list = list()
count_no = list()

text_file = open('test_data.txt', 'r')

# creating the list of individual words from the corpus
myList = (''.join(char if char not in string.punctuation else ' ' +
          char for char in text_file.read())).split(' ')

#creating the list of all the trigrams possible
temp1 = ''
temp2 = '\n'
for temp_word in myList:
    temp_list = [temp1, temp2, temp_word, 1]
    if trigram.count(temp_list) is 0:
        trigram.append(temp_list)
        count_no.append(1)
    else:
        count_no[trigram.index(temp_list)] += 1
    temp1 = temp2
    temp2 = temp_word
for temp in trigram:
    temp[3] = count_no[trigram.index(temp)]

#creating list of most probable trigrams among all possible trigrams
sorted_list = sorted(trigram, key=lambda x: x[3],
                     reverse=True)[0: CORPUS_LENGTH]

#making the word list out of trigrams
for line in sorted_list:
    for i in range(3):
        if not word_list.count(line[i]):
            word_list.append(line[i])
input_list = sorted(word_list)

#creating the vectors
num_words = len(input_list)
inp = np.zeros((num_words, num_words))
for i in range(num_words):
    inp[i][i] = 1
inp1_vec = np.zeros((1, num_words))
inp2_vec = np.zeros((1, num_words))
out_vec = np.zeros((1, num_words))
for temp_list in sorted_list:
    inp1 = word_list.index(temp_list[0])
    inp2 = word_list.index(temp_list[1])
    out = word_list.index(temp_list[2])
    inp1_vec = np.concatenate((inp1_vec, [inp[inp1, :]]), axis=0)
    inp2_vec = np.concatenate((inp2_vec, [inp[inp2, :]]), axis=0)
    out_vec = np.concatenate((out_vec, [inp[out, :]]), axis=0)
inp_vec = np.concatenate((inp1_vec, inp2_vec), axis=1)

#building the dataset
dataset = SupervisedDataSet(2 * num_words, num_words)
for i in range(len(sorted_list) + 1):
    dataset.addSample(inp_vec[i, :], out_vec[i, :])
tstdata, trndata = dataset.splitWithProportion(0.25)

#building the network
net = FeedForwardNetwork()
input_layer = LinearLayer(2 * num_words, name='input_layer')
hidden_layer = TanhLayer(num_words, name='hidden')
output_layer = SigmoidLayer(num_words, name='output_layer')
net.addInputModule(input_layer)
net.addModule(hidden_layer)
net.addOutputModule(output_layer)
net.addConnection(FullConnection(input_layer,
                                 hidden_layer,
                                 name='in_to_hidden'))
net.addConnection(FullConnection(hidden_layer,
                                 output_layer,
                                 name='hidden_to_out'))
net.sortModules()

#backpropagation
trainer = BackpropTrainer(net, dataset=trndata,
                          momentum=0.1,
                          verbose=True,
                          weightdecay=0.01)
#error checking part
for i in range(10):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(), trndata['target'])
    tstresult = percentError(trainer.testOnClassData(dataset=tstdata),
                             tstdata['target'])

trigram_file = open('trigram.txt', 'w')
trigram_file.writelines(["%s\n" % item for item in sorted_list])

word_file = open('word_list', 'w')
word_file.writelines(["%s\n" % item for item in input_list])

word_file.close()
trigram_file.close()
text_file.close()
