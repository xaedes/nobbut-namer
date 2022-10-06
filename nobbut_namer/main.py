
import argparse
import os
import requests
import string
import itertools
import math
import random
import numpy as np
from progress.bar import Bar


def get_user_cache_dir():
    # get user cache directory.
    # if it does not exist, it is created.
    user_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nobbut_namer")
    if not os.path.exists(user_cache_dir):
        os.mkdir(user_cache_dir)
    return user_cache_dir

def main():
    # parse arguments: list of keywords used as seed for name generator, count of names to generate, vocabulary file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", help="list of keywords used as seed for name generator", nargs="+", default=[""])
    parser.add_argument("-n", "--count", help="count of names to generate", type=int, default=10)
    parser.add_argument("-s", "--size", help="number of words in each name", type=int, default=2)
    parser.add_argument("-v", "--vocabulary", help="vocabulary file", type=str, default="vocabulary.txt")
    parser.add_argument("-t", "--top", help="top n words to use", type=int, default=1000)
    parser.add_argument("--sep", help="separator between words in name", type=str, default="-")
    parser.add_argument("--cache", help="cache file", type=str, default="embeddings_cache.npz")
    args = parser.parse_args()

    if len(args.keywords) == 0:
        args.keywords = [""]

    # prepend path of user cache dir to vocabulary and cache file.
    cache_dir = get_user_cache_dir()
    args.vocabulary = os.path.join(cache_dir, args.vocabulary)
    args.cache = os.path.join(cache_dir, args.cache)

    # print arguments left justified with enough padding.
    print("keywords".ljust(15), ", ".join(args.keywords))
    print("count".ljust(15), args.count)
    print("vocabulary".ljust(15), args.vocabulary)
    print("top".ljust(15), args.top)
    print("size".ljust(15), args.size)
    print("sep".ljust(15), args.sep)
    print("cache".ljust(15), args.cache)

    print("Loading cache...")
    #cache = Cache(args.cache)

    print("Loading model...")
    text_model = TextEmbeddingModel()
    get_text_embedding = text_model.get_text_embedding

    # the vocabulary is a list of words that are used to generate names.
    # if it does not exist, a list of words suitable for name generation is downloaded and used to generate the vocabulary.
    # if the vocabulary exists, it is loaded and used to generate names.

    print("Loading vocabulary...")
    vocabulary = get_vocabulary(args.vocabulary)
    
    print("Embedding vocabulary...")
    voc_embeddings = get_vocabulary_embeddings(get_text_embedding, vocabulary, args.cache, text_model.device)
    
    # generate names and print them.
    names = generate_names(
        get_text_embedding,
        vocabulary, voc_embeddings,
        args.keywords, 
        args.count, args.size, args.top, args.sep,
    )

    print("Generated names:")
    for name in names:
        print(name)

def get_vocabulary(filename):
    # get the vocabulary from the file if it exists.
    # if it does not exist, download the vocabulary from the internet and save it to the file specified in filename.
    # return the vocabulary.
    if os.path.exists(filename):
        with open(filename) as f:
            vocabulary = [line.strip() for line in f.readlines()]
    else:
        # download list of words used to generate names.
        print("Downloading list of words used to generate names...")
        download_url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
        response = requests.get(download_url)
        words = response.text.splitlines()

        # filter out words that are too short or too long.
        # words that are too short are not suitable for name generation.
        # words that are too long are not suitable for name generation.
        print("Filtering out words that are too short or too long...")
        words = [word for word in words if 3 <= len(word) <= 10]

        # filter out words that contain non alphabetic characters.
        # words that contain non alphabetic characters are not suitable for name generation.
        print("Filtering out words that contain non alphabetic characters...")
        words = [word for word in words if word.isalpha()]

        # filter out words that contain numbers.
        # words that contain numbers are not suitable for name generation.
        print("Filtering out words that contain numbers...")
        words = [word for word in words if not any(char.isdigit() for char in word)]

        # filter out words that contain special characters.
        # words that contain special characters are not suitable for name generation.
        print("Filtering out words that contain special characters...")
        words = [word for word in words if not any(char in string.punctuation for char in word)]

        # filter out words that contain non ascii characters.
        # words that contain non ascii characters are not suitable for name generation.
        print("Filtering out words that contain non ascii characters...")
        words = [word for word in words if all(ord(char) < 128 for char in word)]

        # filter out words that contain only one character.
        # words that contain only one character are not suitable for name generation.
        print("Filtering out words that contain only one character...")
        words = [word for word in words if len(word) > 1]

        # filter out words that are too similar.
        # words that are too similar are not suitable for name generation.
        # this is a long process as it has quadratic complexity.
        # use python package "progress" to show progress while processing.
        # from progress.bar import Bar
        # words = list(Bar("Filtering out words that are too similar...", max=len(words)*len(words)).iter(
        #     (word for word in words if not any(word in other_word for other_word in words if word != other_word))
        # ))

        # remove words that are too similar to each other.
        # words that are too similar to each other are not suitable for name generation.
        print("Removing words that are too similar to each other...")
        words = list(set(words))

        # sort words alphabetically.
        print("Sorting words alphabetically...")
        words.sort()

        # write words to file.
        print("Writing words to file...")
        with open(filename, "w") as f:
            f.write("\n".join(words))

        # return words.
        vocabulary = words
    
    return vocabulary
    
def get_vocabulary_embeddings(get_text_embedding, vocabulary, cache_filename, device, batch_size=128):
    try:
        # try to load vocabulary embeddings from cache.
        res = np.load(cache_filename)
        voc_embeddings = res["voc_embeddings"]
        if voc_embeddings.shape[0] != len(vocabulary):
            raise ValueError("vocabulary size mismatch")
        # convert to torch tensor.
        voc_embeddings = torch.from_numpy(voc_embeddings).to(device)
        return voc_embeddings
    except:
        # split vocabulary into batches
        voc_batches = [vocabulary[i:min(i+batch_size,len(vocabulary))] for i in range(0, len(vocabulary), batch_size)]
        if len(voc_batches) > 0 and len(voc_batches[-1]) < batch_size:
            voc_batches[-1] += [""] * (batch_size - len(voc_batches[-1]))
        # use batches to generate embeddings.
        voc_embeddings = list(Bar(max=len(voc_batches)).iter(map(get_text_embedding, voc_batches)))
        voc_embeddings = torch.stack(voc_embeddings)
        voc_embeddings = voc_embeddings.reshape(-1,voc_embeddings.shape[-1])
        voc_embeddings = voc_embeddings[:len(vocabulary)]
        np.savez(cache_filename, voc_embeddings=voc_embeddings.cpu().numpy())
        return voc_embeddings


# use a pretrained pytorch model from huggingface to generate text embeddings.
# the model is initialized when the class TextEmbeddingModel is instantiated.
# the pretrained model is stored in a cache folder and automatically downloaded 
# if it does not exist.
# we use huggingface for the pretrained model which requires no authentication token.
# using the function TextEmbeddingModel.get_text_embedding(self, text) the embedding 
# of a text can be generated.

# imports for text embedding model.
import torch
from transformers import AutoModel, AutoTokenizer

class TextEmbeddingModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device:", self.device)
        self.model.to(self.device)

    def get_text_embedding(self, text):
        # generate a text embedding from the text.
        # the text is tokenized using the tokenizer of the pretrained model.
        # the tokens are converted to the model's input format.
        # the tokens are passed to the model to generate the text embedding.
        # the text embedding is returned.
        if type(text) == list:
            tokenized_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        else:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
        tokenized_text = {key: tokenized_text[key].to(self.device) for key in tokenized_text}
        with torch.no_grad():
            text_embedding = self.model(**tokenized_text).pooler_output
        return text_embedding

def generate_names(get_text_embedding, vocabulary, voc_embeddings, keywords, count, size, top, sep):
    with torch.no_grad():
        # generate names using the keyword as seed.
        # the names are generated using the text embedding of the keyword.
        # the text embedding is used to find the nearest words in the vocabulary.

        print("embedding keywords...")
        text_embeddings = list(Bar(max=len(keywords)).iter(map(get_text_embedding, keywords)))
        #text_embeddings = torch.stack(text_embeddings)
        #print("text_embeddings.shape:", text_embeddings.shape)

        
        # generate word tuples from the vocabulary by picking two words from the vocabulary.
        # sort vocabulary by minimum similarity to on of the keywords and randomly pick from the top 100.
        # use compare_embeddings() to compare the similarity of text embeddings.
        def compare_embeddings(a, b):
            # compare by similarity of two text embeddings.
            # the embeddings are compared using cosine similarity.
            # the similarity is returned.
            return 1-torch.nn.functional.cosine_similarity(a, b).item()

        def compare_embeddings_multi(a, b_multi):
            # compare by similarity between a text embedding and multiple text embeddings.
            # the embeddings are compared using cosine similarity.
            # the combined similarity is returned.
            return torch.nn.functional.cosine_similarity(a, b_multi)
        
        
        # find top words in vocabulary that are most similar to on of the keywords.
        print("score similarities of vocabulary to keywords...")
        similarities = list(Bar(max=len(text_embeddings)).iter((
            compare_embeddings_multi(
                text_embedding.reshape(text_embedding.shape[1]), 
                voc_embeddings
            )
            for text_embedding in text_embeddings
        )))
        
        # convert list of tensors to tensor
        similarities = torch.stack(similarities)
        similarities = similarities.mean(dim=0)

        # convert tensor to numpy array
        similarities = similarities.cpu().numpy()

        top_words = sorted(
            zip(vocabulary,similarities), 
            key=lambda word_sim: word_sim[1]
        )[::-1][:top]
        top_word_similarities = [word_sim[1] for word_sim in top_words]
        top_words = [word_sim[0] for word_sim in top_words]
        print("generating names...")
        # generate names by randomly picking two words from the top 100 words.
        combinations_limit = top*2
        names = []
        for words in random_combinations(top_words, top_word_similarities, size, count, combinations_limit):
            names.append(sep.join(words))
        
        # return names.
        return names

def random_combinations(values, weights, r, count, combinations_limit):
    # generate a list of random combinations of length r from the iterable.
    # the list contains count combinations.
    # the combinations are generated using the python package "itertools".
    # the combinations are returned.
    
    # compute the number of combinations.
    num_combinations = int(math.factorial(len(values)) / (math.factorial(r) * math.factorial(len(values)-r)))
    combinations = list(Bar(max=num_combinations).iter(
        itertools.combinations(range(len(values)), r),
    ))
    if count < len(combinations):
        # randomly pick combinations from the weighted list of combinations.
        combinations_weights = [
            np.product([weights[i] for i in combination])
            for combination in combinations
        ]
        combinations = random.choices(combinations, weights=combinations_weights, k=count)
    
    return [
        tuple(values[i] for i in combination)
        for combination in combinations
    ]

def list_atmost(iterable, count):
    # return a list of at most count elements from the iterable.
    # the list is returned.
    return list(itertools.islice(iterable, count))
