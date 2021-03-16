import re
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
import numpy as np
from tensorflow import keras
import random
import matplotlib.pyplot as plt


def loaddata(data):
    with open(data, encoding='utf-8') as f:
        Corpus = f.readlines()  
    Corpus = ' '.join(Corpus).lower().split('\n')
    return Corpus


class tensortext:
    def __init__(self, data):
        self.corpus = data
        #self.column = column
    
    ran_list = ['my love', 'sweet heart', 'honey sweet', 'justine love', 'to me']
    # random item from list
    pick_ran = random.choice(ran_list)
    '''def cleanText(self):
        'Removes emojis and hashes'
        emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                "]+", re.UNICODE)
    
        text = re.sub(emoji, '', self.data)
        text = re.sub(r'#\w+', ' ', text)
        self.corpus = text
        return self.corpus'''
        
        
    def sequence(self, padding_method):
        '''Tokenizes the data and turns the tokens into sequences'''
        self.padding_method = padding_method
        print('Tokenizing your data', '-------'*7)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.corpus)
        totalWords = len(self.tokenizer.word_index) + 1
    
        self.sequences = []
        print('padding sequence', '-------'*7)
        for line in self.corpus:
            tokenList = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(tokenList)):
                ngramSequence = tokenList[:i+1]
                self.sequences.append(ngramSequence)
        #self.sequences = sequences
        self.totalwords = totalWords
        #print(totalWords)
        #print(self.sequences)
        '''Gives the sequences a uniform length by padding them'''
    
        self.maxSequenceLen = max([len(seq) for seq in self.sequences])
        self._sequences = np.array(pad_sequences(self.sequences, maxlen=self.maxSequenceLen, padding=self.padding_method))
    
        self.predictors, self.label = self._sequences[:,:-1], self._sequences[:,-1]
        self._label = to_categorical(self.label, num_classes=self.totalwords)
        #print(totalWords)
        #print(sequences[:5])
        return self.predictors, self._label, self.maxSequenceLen, self.totalwords
    
    def configmodel(self,seq_text, lstmlayer, activation):
    
        '''Configures the neural network'''
        self.predictors, self._label, self.maxSequenceLen, self.totalwords = seq_text
        if lstmlayer and activation == None:
            lstmlayer = 128
            activation = 'softmax'
        model = models.Sequential()
        model.add(layers.Embedding(self.totalwords, 64,input_length=self.maxSequenceLen - 1))
        model.add(layers.LSTM(lstmlayer))
        model.add(layers.Dense(self.totalwords, activation=activation))
        self.model = model
    
        print(self.model.summary())
        return self.model
    
    def fit(self,loss, optimizer, metrics, epochs, verbose, patience):
        #self.predictors, self._label, self.maxSequenceLen, self.totalwords = seq_data
        
    
        '''Sets the training parameters and fits the model to the data'''
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        history = self.model.fit(self.predictors, self._label, epochs=epochs, batch_size= batch, verbose = verbose,
                        callbacks=[EarlyStopping(monitor='loss', patience=patience,
                                                 restore_best_weights=True)])
        self.history = history
        return self.history
    
    def predict(self, sample_text):               #A text seed is provided
    
        '''Predicts the next text sequences'''
        #model = self.model    
        for wordLength in range(50):   #Generates a text with a range of word length
            tokenList = self.tokenizer.texts_to_sequences([sample_text])[0]  #Turns the seed into sequences
            tokenList = pad_sequences([tokenList], maxlen=self.maxSequenceLen - 1, padding=self.padding_method)
            predicted = self.model.predict_classes(tokenList, verbose=verbose) #Predicts the next sequence(generated
            outputWord = " "                                         #text)  
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    outputWord = word
                    break
            sample_text += " " + outputWord
            #Returns the seed plus generated text
        print(sample_text)
        return sample_text
    
    def plot_loss_accuracy(self):
        '''Visualizes the performance of the model in-between epochs'''
        accuracy = self.history.history['accuracy']
        loss = self.history.history['loss']
        epochs = range(1, len(accuracy) + 1)

        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        ax1.set(title='Language Model Accuracy', ylabel='Accuracy')
        ax2.set(title='Language Model Loss', xlabel='Epochs', ylabel='Loss')
        plot = ax1.plot(epochs, accuracy, 'bo', label='Accuracy')
        plot = ax2.plot(epochs, loss, 'bo', label='Loss')

        fig.suptitle('Loss/Accuracy of the Language Model', fontsize=16, fontweight = 'bold')
    
        return plot









