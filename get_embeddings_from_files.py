import io
import torch
from embedding import Embedding 
print("---EN-DE---\n")

# Load Files
with io.open("./data_en_de/train.ende.src", "r", encoding="utf8") as ende_src:
    english = ende_src.read().splitlines()    
with io.open("./data_en_de/train.ende.mt", "r", encoding="utf8") as ende_mt:
    german = ende_mt.read().splitlines()
with io.open("./data_en_de/dev.ende.src", "r", encoding="utf8") as ende_src:
    english_val = ende_src.read().splitlines()
with io.open("./data_en_de/dev.ende.mt", "r", encoding="utf8") as ende_mt:
    german_val = ende_mt.read().splitlines()
with io.open("./data_en_de/test.ende.src", "r", encoding="utf8") as ende_src:
    english_test = ende_src.read().splitlines()
with io.open("./data_en_de/test.ende.mt", "r", encoding="utf8") as ende_mt:
    german_test = ende_mt.read().splitlines()
    
lemmatize = True

# Get and save English Embeddings

en_emb = Embedding('en')

en_embeddings_max, en_embeddings_mean = en_emb.get_batch_embedding(
    english, batch_size=10, lemmatize=lemmatize)

torch.save(en_embeddings_max, 
           './english_train_embeddings_max.pt')
torch.save(en_embeddings_mean, 
           './english_train_embeddings_mean.pt')

en_val_embeddings_max, en_val_embeddings_mean = en_emb.get_batch_embedding(
    english_val, batch_size=10, lemmatize=lemmatize)

torch.save(en_val_embeddings_max, 
           './english_val_embeddings_max.pt')
torch.save(en_val_embeddings_mean, 
           './english_val_embeddings_mean.pt')

en_test_embeddings_max, en_test_embeddings_mean = en_emb.get_batch_embedding(
    english_test, batch_size=10, lemmatize=lemmatize)

torch.save(en_test_embeddings_max, 
           './english_test_embeddings_max.pt')
torch.save(en_test_embeddings_mean, 
           './english_test_embeddings_mean.pt')


# Get and save German Embeddings

de_emb = Embedding('de')

de_embeddings_max, de_embeddings_mean  = de_emb.get_batch_embedding(
    german, batch_size=10, lemmatize=lemmatize)

torch.save(de_embeddings_max, 
           './german_embeddings_max.pt')
torch.save(de_embeddings_mean, 
           './german_embeddings_mean.pt')

ge_val_embeddings_max, ge_val_embeddings_mean = de_emb.get_batch_embedding(
    german_val, batch_size=10, lemmatize=lemmatize)

torch.save(ge_val_embeddings_max, 
           './german_val_embeddings_max.pt')
torch.save(ge_val_embeddings_mean, 
           './german_val_embeddings_mean.pt')

ge_test_embeddings_max, ge_test_embeddings_mean = de_emb.get_batch_embedding(
    german_test, batch_size=10, lemmatize=lemmatize)

torch.save(ge_test_embeddings_max, 
           './german_test_embeddings_max.pt')
torch.save(ge_test_embeddings_mean, 
           './german_test_embeddings_mean.pt')


