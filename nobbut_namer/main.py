
import argparse
import os
import requests
import string
import itertools
import pickle
import random
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
    parser.add_argument("--cache", help="cache file", type=str, default="embeddings_cache.pkl")
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


    cache = Cache(args.cache)

    # generate names and print them.
    print("Generated names:")
    for name in generate_names(cache, args.keywords, args.count, args.size, args.vocabulary, args.top, args.sep):
        print(name)


class Cache:
    def __init__(self, filename):
        self.filename = filename
        self.cache = {}
        
        # load cached embeddings from file if they exist.
        # use pickle
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.cache = pickle.load(f)

    def save(self):
        with open(self.filename, "wb") as f:
            pickle.dump(self.cache, f)

    def cached(self, fn):
        # cache the result of a function call.
        # the function call is cached using the function parameters as key.
        # the function call is only executed if the result is not cached.
        # the result of the function call is returned.
        cache = self.cache
        def wrapper(*args):
            if args not in cache:
                cache[args] = fn(*args)
            return cache[args]
        return wrapper

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

# use a pretrained pytorch model from huggingface to generate text embeddings.
# the model is initialized when the class TextEmbeddingModel is instantiated.
# the pretrained model is stored in a cache folder and automatically downloaded 
# if it does not exist.
# we use huggingface for the pretrained model which required no authentication token.
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
        tokenized_text = self.tokenizer(text, return_tensors="pt")
        tokenized_text = {key: tokenized_text[key].to(self.device) for key in tokenized_text}
        with torch.no_grad():
            text_embedding = self.model(**tokenized_text).pooler_output
        return text_embedding

def generate_names(cache, keywords, count, size, vocabulary_filename, top, sep):
    # generate names using the keyword as seed.
    # the names are generated using the text embedding of the keyword.
    # the text embedding is used to find the nearest words in the vocabulary.
    # the vocabulary is a list of words that are used to generate names.
    # if it does not exist, a list of words suitable for name generation is downloaded and used to generate the vocabulary.
    # if the vocabulary exists, it is loaded and used to generate names.
    print("loading model...")
    text_model = TextEmbeddingModel()
    get_text_embedding = cache.cached(text_model.get_text_embedding)

    print("loading vocabulary...")
    vocabulary = get_vocabulary(vocabulary_filename)
    
    print("embedding vocabulary...")
    voc_embeddings = list(Bar(max=len(vocabulary)).iter(map(get_text_embedding, vocabulary)))
    
    print("embedding keywords...")
    text_embeddings = list(Bar(max=len(keywords)).iter(map(get_text_embedding, keywords)))
    text_embeddings = torch.stack(text_embeddings)
    cache.save()
    names = []
    
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
        return torch.nn.functional.cosine_similarity(a, b_multi).mean().item()
    
    print("generating names...")
    # find top words in vocabulary that are most similar to on of the keywords.
    similarities = [
        compare_embeddings_multi(voc_embedding, text_embeddings)
        for voc_embedding in voc_embeddings
    ]
    top_words = sorted(
        zip(vocabulary,similarities), 
        key=lambda word_sim: word_sim[1]
    )[::-1][:top]
    top_word_similarities = [word_sim[1] for word_sim in top_words]
    top_words = [word_sim[0] for word_sim in top_words]
    # generate names by randomly picking two words from the top 100 words.
    for words in random_combinations(top_words, top_word_similarities, size, count):
        names.append(sep.join(words))
    
    # return names.
    return names

def random_combinations(values, weights, r, count):
    # generate a list of random combinations of length r from the iterable.
    # the list contains count combinations.
    # the combinations are generated using the python package "itertools".
    # the combinations are returned.
    combinations = list(itertools.combinations(range(len(values)), r))
    if count < len(combinations):
        # randomly pick combinations from the weighted list of combinations.
        combinations_weights = [
            max(weights[i] for i in combination)
            for combination in combinations
        ]
        combinations = random.choices(combinations, weights=combinations_weights, k=count)
    
    return [
        tuple(values[i] for i in combination)
        for combination in combinations
    ]

