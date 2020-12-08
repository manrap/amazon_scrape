from collections import defaultdict
import math
import csv
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

# vectorizer = TfidfVectorizer()
#
# df = pd.read_csv("amazon_products.tsv", sep="\t", header=0)
# size = df.size
# df.fillna("nessuna descrizione", inplace=True)
#
# print(df.head())
#
# X = vectorizer.fit_transform(df['description']) # Store tf-idf representations of all docs
# print(X.shape)
# query = "computer lenovo"
# query_vec = vectorizer.transform([query]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
# results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc
# for i in results.argsort()[-10:][::-1]:
#     print(df.iloc[i,0],"--",df.iloc[i,1])

stop_words = set(stopwords.words('italian'))
prod_dict = defaultdict()
index = defaultdict()
df_dict = defaultdict()
tf_dict = defaultdict()
tf_dict_doc = defaultdict()

prod_number = 1
with open("amazon_products.tsv", "r", encoding="utf-8") as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter="\t")
    # row_count = sum(1 for row in tsv_reader)
    # print(row_count)
    for row in tsv_reader:
        product_name = row[0]
        product_url = row[1]
        product_description = row[3]

        description_tokens = word_tokenize(product_description)
        # if description is present use it
        if len(description_tokens) > 0:
            words = [token.lower() for token in description_tokens if token.isalpha()]
            filtered_sentence = [w for w in words if not w in stop_words]
            stemmed_sentence = [PorterStemmer().stem(w) for w in filtered_sentence]
            prod_dict[prod_number] = filtered_sentence
        else:
            # else use the name
            name_tokens = word_tokenize(product_name)
            words = [token.lower() for token in name_tokens if token.isalpha()]
            filtered_sentence = [w for w in words if not w in stop_words]
            stemmed_sentence = [PorterStemmer().stem(w) for w in filtered_sentence]
            prod_dict[prod_number] = filtered_sentence

        # Document Frequency
        unique_stems = set(stemmed_sentence)
        for unique_stem in unique_stems:
            if not unique_stem in df_dict.keys():
                df_dict[unique_stem] = 1
            else:
                df_dict[unique_stem] += 1

            if not unique_stem in tf_dict.keys():
                freq = stemmed_sentence.count(unique_stem)
                tf_dict[unique_stem] = [(prod_number, freq / len(stemmed_sentence))]
            else:
                freq = stemmed_sentence.count(unique_stem)
                tf_dict[unique_stem].append((prod_number, freq / len(stemmed_sentence)))

            if not prod_number in tf_dict_doc.keys():
                freq = stemmed_sentence.count(unique_stem)
                tf_dict_doc[prod_number] = [(unique_stem, freq / len(stemmed_sentence))]
            else:
                freq = stemmed_sentence.count(unique_stem)
                tf_dict_doc[prod_number].append((unique_stem, freq / len(stemmed_sentence)))
            # freq = stemmed_sentence.count(unique_stem)
            # tf_dict[unique_stem][prod_number] = freq / len(stemmed_sentence)

        for stem in stemmed_sentence:
            if not stem in index.keys():
                freq = stemmed_sentence.count(stem)
                tf = freq / len(stemmed_sentence)
                index[stem] = [(prod_number, freq)]
                # counter = counter + len(stem) + 1
            else:
                freq = stemmed_sentence.count(stem)
                index[stem].append((prod_number, freq))
                # counter = counter + len(stem) + 1

        prod_number += 1
        # print(filtered_sentence)

# print(index)
print(prod_number)
# print(tf_dict_doc.keys())
docs = len(tf_dict_doc)
print(docs)
# print(df_dict)
idf_dict = defaultdict()
for term, df in df_dict.items():
    idf_dict[term] = math.log(prod_number / df)

tf_idf_dict = defaultdict()
# print(idf_dict)
for term, list in tf_dict.items():
    for tf_item in list:
        tf_idf = tf_item[1] * idf_dict[term]
        if not term in tf_idf_dict.keys():
            tf_idf_dict[term] = [(tf_item[0], tf_idf)]
        else:
            tf_idf_dict[term].append((tf_item[0], tf_idf))

# print(tf_idf_dict)

# print(tf_dict_doc[101])
tf_idf_dict_doc = defaultdict()

for term, list in tf_idf_dict.items():
    for tf_item in list:
        if not tf_item[0] in tf_idf_dict_doc.keys():
            tf_idf_dict_doc[tf_item[0]] = [(term, tf_idf)]
        else:
            tf_idf_dict_doc[tf_item[0]].append((term, tf_idf))


# # print(idf_dict)
# for prod_num, list in tf_dict_doc.items():
#     for tf_item in list:
#         tf_idf = tf_item[1] * idf_dict[tf_item[0]]
#         if not prod_num in tf_idf_dict.keys():
#             tf_idf_dict_doc[prod_num] = [(tf_item[0], tf_idf)]
#         else:
#             tf_idf_dict_doc[prod_num].append((tf_item[0], tf_idf))

# print(tf_idf_dict_doc)
# query esempio doc 84


def cosine_sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


unique_word_count = len(df_dict)
print("unique words: " + str(unique_word_count))

doc_tf_idf_vector = np.zeros((docs, unique_word_count))
total_vocab = [key for key in df_dict]

for doc, list in tf_idf_dict_doc.items():
    # try:
    for token in list:
        print(token)
        print(df_dict[token])
        ind = total_vocab.index(token)  #df_dict è come se non avesse delle parole in tf_idf_dict
        print(ind)
        print(tf_idf_dict_doc[doc, token])
        # doc_tf_idf_vector[doc][ind] = tf_idf_dict_doc[doc, token]
        # print(doc_tf_idf_vector[doc][ind])
    # except:
    #     pass


def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))

    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}

    for token in np.unique(tokens):

        tf = counter[token] / words_count
        # df = doc_freq(token)
        df = df_dict.get(token)
        if df is None:
            idf = 0
        else:
            idf = math.log((docs + 1) / (df + 1))

        try:
            term_index = total_vocab.index(token)
            Q[term_index] = tf * idf
        except:
            pass
    return Q


def cosine_similarity(k, query):
    print("Cosine Similarity")
    # preprocessed_query = preprocess(query)
    # tokens = word_tokenize(str(preprocessed_query))

    query_tokens = word_tokenize(query)
    words = [token.lower() for token in query_tokens if token.isalpha()]
    filtered_query = [w for w in words if not w in stop_words]
    tokens = [PorterStemmer().stem(w) for w in filtered_query]

    print("\nQuery:", query)
    print("")
    print(tokens)

    d_cosines = []

    query_vector = gen_vector(tokens)
    for d in doc_tf_idf_vector:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]

    print("")

    print(out)


#     for i in out:
#         print(i, dataset[i][0])

query = "custodia tablet"

cosine_similarity(10, query)
