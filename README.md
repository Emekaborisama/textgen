
<h1 align="center">
<p>Text Gen :goat:</p>

<p align="center">

| Total Downloads | [![Downloads](https://pepy.tech/badge/text-gen)](https://pepy.tech/project/text-gen)
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.3.0-orange?logo=tensorflow">
<a href="https://pypi.org/project/text-gen/">
  <img alt="PyPI" src="https://img.shields.io/pypi/v/text-gen?color=%234285F4&label=release&logo=pypi&logoColor=%234285F4">
</a>
</p>
</h1>
<h2 align="center">
<p>Almost State-of-the-art Text Generation library</p>
</h2>

<p align="center">
Text gen is a python library that allow you build a custom text generation model with ease :smile:
 Something sweet built with Tensorflow and Pytorch(coming soon) - This is the brain of Rosalove ai (https://rosalove.xyz/)</p>



## How to use it
Install text-gen
```bash
pip install -U text-gen
```
import the library
```python
from text_gen import ten_textgen as ttg
```


Load your data. your data must be in a text format.

Download the example data from the [example folder](https://github.com/Emekaborisama/textgen/tree/master/example)
#### load data
```python
data = 'rl.csv'
text = ttg.loaddata(data)
```



```python
pipeline = ttg.tentext(text)
seq_text = pipeline.sequence(padding_method = 'pre')
configg = pipeline.configmodel(seq_text, lstmlayer = 128, activation = 'softmax', dropout = 0.25)

```


#### train model
```python
model_history = pipeline.fit(loss = 'categorical_crossentropy', optimizer = 'adam', batch = 300, metrics = 'accuracy', epochs = 500, verbose = 0, patience = 10)

```


#### generate text using the phrase
```python
pipeline.predict('hello love', word_length = 200, segment = True)
```


#### plot loss and accuracy
```python
pipeline.plot_loss_accuracy()
```

#### Hyper parameter optimization
Tune your model to know the best optimizer, activation method to use.
```python
pipeline.hyper_params(epochs = 500)
```

<h1 align="center">
<span> Give us a star :star: </span> 🐉
</h1>

If you want to contribute, take a look at the issues and the [Futurework.md](https://github.com/Emekaborisama/textgen/blob/master/futurework.md) file 


Contributors 

- [Emeka Boris Ama](https://twitter.com/emeka_boris)
- [King Zikie](https://twitter.com/kingzikie)
- [David](https://twitter.com/iEphods)

