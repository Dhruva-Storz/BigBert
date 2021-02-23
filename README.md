# BigBert

This project was done as part of the Natural Language Processing module at Imperial 
College London. The task involves the implementation of a regression model that evaluates 
the quality of a machine translated sentence between English and German as part of the challenge below:

https://competitions.codalab.org/competitions/22831?secret_key=253ffaba-e3ea-44b3-a936-e67e372fa60c

Our models came in third place in the mean absolute error (MAE) metric
 
## Getting Started

These instructions will get you a copy of the project up and running on your local 
machine for development and testing purposes. 

Note: The original data and embeddings used are not available on the github repository due to memory constraints.

# Run pipeline

Below are intructions of how to run the pipeline (we do not create the embeddings from scratch in this format. If you want to create the embeddings see the Get Embeddings format):
* Download the whole directory 
* Install the requirements
* If you want to create embeddings for a new dataset use ```get_embeddings_from_files.py```, otherwise you can work with our already saved embeddings for our current data
* Load embeddings in ```main.py```
* Set the hyperparameters in ```main.py``` and create a model
* Run and get results

 

### Requirements

```
transformers
sklearn
scipy
torch
numpy
nltk
```

### Example of how to get embeddings

How to get embeddings for:
```
data = ['You can say to me.',
        "I love machine learning It's awesome",
        "I love coding in python",           
        "I love building chatbots",
        "they chat amagingly well",
        'Hello Mr. Smith, how are you doing today?', 
        'The weather is great, and city is awesome.', 
        'The sky is pinkish-blue.', 
        "You shouldn't eat cardboard",
        'Hello beautiful world']
```
Create an encoder:

```
import Embedding from embedding

english_encoder = Embedding('en')
#Loading English BERT tokenizer...

#Loading English BERT Model...
```

Get the embeddings:

```
batch_size = 5
lemmatize = True
remove_stop_words = False

embeddings_max, embeddings_mean = english_encoder.get_batch_embedding(
                                data, batch_size = batch_size,lemmatize=lemmatize, 
                                remove_stop_words=remove_stop_words)
#Processing batch 1...
#Getting embedding for batch 1...

#Processing batch 2...
#Getting embedding for batch 2...

#DONE!
#Embedding size is (10,768)

```
### An example of how to load Saved Embeddings

```
import torch

# Load Embeddings

english_max = torch.load('./Embeddings/english_train_embeddings_max.pt')
german_max = torch.load('./Embeddings/german_embeddings_max.pt')

english_avg = torch.load('./Embeddings/english_train_embeddings_mean.pt')
german_avg = torch.load('./Embeddings/german_embeddings_mean.pt')

# Load scores/labels

f_train_scores = open("./data_en_de/train.ende.scores",'r')
de_train_scores = f_train_scores.readlines()

```
Create feature vectors
```
english = torch.cat((english_max, english_avg), dim=1)
german = torch.cat((german_max, german_avg), dim=1)

en_ge_cat = torch.cat((english,german), dim=1)
en_ge_product = english * german
en_ge_abs_dif = (english - german).abs()

# Tensor of Shape (7000, 768 x 8)
# (u', v' |u' - v'|, u' * v')
X_train = torch.cat((en_ge_cat, en_ge_product, en_ge_abs_dif), dim=1)

y_train = np.array(de_train_scores).astype(float)

```

### Train a model

Choose a mode from {```'SVR', 'MLP_torch', 'MLP_sckit'```} and execute ```main.py```.
You can define all the hyperparameters in the ```main()``` fucntion.

### Relevant papers

[Regression for machine translation evaluation at the sentence level](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s10590-008-9046-1.pdf&casa_token=hfknXfFo7osAAAAA:aaf-7G6ynHdGYxhhzuNvED0qNOmfK5UdgwPK-cCP4iwk0RY-J1svV-k7Juhwdvysyb8rWK36deqGgoxZJQ)

[Ruse: Regressor using sentence embeddings for automatic machine translation evaluation](https://www.aclweb.org/anthology/W18-6456.pdf)

[Putting evaluation in context: Contextual embeddings improve machine translation evaluation](https://www.aclweb.org/anthology/P19-1269.pdf)

### Authors

* **Christos Seas** 
* **Dhruva Gowda Storz** 
* **George Yiasemis**


