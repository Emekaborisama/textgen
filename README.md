
<h1 align="center">
<p>Text Gen :goat:</p>

<p align="center">

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
 Something sweet built with Tensorflow and Pytorch - This is the brain of rosalove ai(rosalove.xyz)

</p>


## How to use it


```bash
pip install -U text-gen
```

```python
from text_gen import ten_textgen as ttg
```
#### load data
```python
data = 'rosalove.csv'
text = ttg.loaddata(data)
```


#### parameters
```python 
activation = 'softmax'
lstmlayer = 128
padding_method = 'pre'

loss='categorical_crossentropy'
optimizer='adam'
metrics='accuracy'
epochs=100
verbose = 0
patience = 10
batch_size = 300

```


```python
pipeline = ttg.tentext(text)
seq_text = pipeline.sequence(padding_method)
configg = pipeline.configmodel(seq_text, lstmlayer, activation)

```


#### train model
```python
model_history = pipeline.fit(loss = loss, optimizer = optimizer, batch_size = batch_size, metrics = metrics, epochs = epochs, verbose = verbose, patience = patience)

```


#### generate text using the phrase
```python
pipeline.predict('hello love')
```


#### plot loss and accuracy
```python
pipeline.plot_loss_accuracy()
```


<h1 align="center">
<span> Give us a star :star: </span> 🐉
</h1>


Contributors 

- [Emeka Boris Ama](https://twitter.com/emeka_boris)
- [King Zikie](https://twitter.com/kingzikie)
- [David](https://twitter.com/iEphods)
