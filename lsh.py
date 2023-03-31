# This is the code for the LSH project of TDT4305

import configparser
import hashlib
import itertools  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import pandas as pd
import random
import numpy as np

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
# the main path where all the data directories are
data_main_directory = Path('data')
# dictionary that holds the input parameters, key = parameter name, value = value
parameters_dictionary = dict()
# dictionary of the input documents, key = document id, value = the document
document_list = dict()


# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(
                docs_Sets[i], docs_Sets[j])

    return similarity_matrix

# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them


def k_shingles():
    docs_k_shingles = []  # holds the k-shingles of each document
    # implement your code here

    # Get the value k from the parameters dictionary
    k = parameters_dictionary.get("k")

    for key in document_list:
        document = document_list[key]
        words = document.split()
        shingles_in_doc = []

        for i in range(len(words)-k+1):
            shingle = words[i:i + k]
            shingle = ' '.join(shingle)

            if shingle not in shingles_in_doc:
                shingles_in_doc.append(shingle)

        docs_k_shingles.append(shingles_in_doc)

    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(shingles):

    docs_signature_sets = []

    for key in range(0, len(document_list)):

        shingle = shingles[key]

        signature_set_shingle = set()

        for i in range(len(shingles[key])):
            hash_val = hash(shingle[i])

            if hash_val not in signature_set_shingle:
                signature_set_shingle.add(hash_val)

        docs_signature_sets.append(signature_set_shingle)
    return docs_signature_sets


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations

# Method to check if a number is prime
def is_prime(n):
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n % f == 0:
            return False
        if n % (f+2) == 0:
            return False
        f += 6
    return True


random.seed(11)


def minHash(document_vector):
    num_permutations = parameters_dictionary.get("permutations")
    num_documents = len(document_list)

    signatures = np.full((num_documents, num_permutations), np.inf)

    # Nye

    for i in range(num_documents):
        a = random.randint(1, 400)
        b = random.randint(1, 400)
        cont = True
        while cont:
            # Generate a random number in the range [lower_bound, upper_bound]
            p = random.randint(400, 1000)

            # Check if the number is prime
            if is_prime(p):
                cont = False

        # nye
        for j in range(num_permutations):
            # gammel
            # for j in range(num_documents):

            for sig in document_vector[j]:

                hash_value = (a * sig + b) % p

                if hash_value < signatures[i][j]:
                    signatures[i][j] = hash_value
    return signatures

# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents


def lsh(m_matrix):

    candidates = []  # list of candidate sets of documents for checking similarity

    buckets = parameters_dictionary.get("buckets")

    bucket_dict = {}
    for i in range(buckets):
        bucket_dict[i] = set()

    # Hash each column of the signature matrix into a bucket
    for j in range(m_matrix.shape[0]):
        hash_val = hash(tuple(m_matrix[j, :]))
        bucket_idx = hash_val % buckets
        bucket_dict[bucket_idx].add(j)

    # Find candidate similar documents from the buckets
    candidates = set()
    for bucket in bucket_dict.values():
        if len(bucket) > 1:
            pairs = list(itertools.combinations(bucket, 2))

            candidates.update(pairs)

    return candidates


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_matrix = []

    for pair in candidate_docs:

        index1 = pair[0]
        index2 = pair[1]
        jac = jaccard(set(min_hash_matrix[index1]), set(
            min_hash_matrix[index2]))
        similarity_matrix.append([index1, index2, jac])

    # implement your code here
    return similarity_matrix

# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity


def return_results(lsh_similarity_matrix):
    document_pairs = []
    t = parameters_dictionary['t']
    count = 0
    for pair in lsh_similarity_matrix:
        threshold = pair[2]
        if threshold > t:
            count += 1
            id1 = pair[0]
            id2 = pair[1]
            # print("Id1: ", id1, " Id2: ", id2, " similarity: ", threshold)
            document_pairs.append([id1, id2])

    # implement your code here
    print("Total above threshold ", t, " :", count)
    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    false_negatives = 0
    false_positives = 0
    # implement your code here

    t = parameters_dictionary['t']
    starttime = time.time()
    lshpairs = 0
    for pair in lsh_similarity_matrix:
        lshpairs += 1
        if (pair[2] > t) and (pair not in naive_similarity_matrix):
            false_positives += 1
    midtime = time.time()
    naivepairs = 0
    for pair in naive_similarity_matrix:
        naivepairs += 1
        if (pair not in lsh_similarity_matrix):
            false_negatives += 1
    endtime = time.time()
    total_positives = len(lsh_similarity_matrix)
    """ print("\nLSH pairs checked: ", lshpairs,
          "\nNaive pairs checked: ", naivepairs)
    print("LSH computation time: ", (midtime - starttime),
          "\nNaive computation time: ", (endtime - midtime)) """

    print("\n\nTotal positives: ", total_positives)
    print("\"True\" positives = ", total_positives-false_positives, "\n")

    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    timeStart = time.time()

    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = minHash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    print("Number of buckets: ", parameters_dictionary.get("buckets"))
    print("Number of candidate document pairs: ", len(candidate_docs))
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(
        candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ",
          parameters_dictionary['t']*100, "% similarity...")
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("\nThe pairs of documents are:\n")
    for p in pairs:
        print(p)
    print("\n")

    # Count false negatives and positives
    if parameters_dictionary['naive']:
        print("Starting to calculate the false negatives and positives...")
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(
            lsh_similarity_matrix, naive_similarity_matrix)
        t17 = time.time()
        print("False negatives = ", false_negatives,
              "\nFalse positives = ", false_positives, "\n\n")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")
    timeEnd = time.time()
    print("Total time for entire program: ", timeEnd-timeStart, "sec\n\n")
