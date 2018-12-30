import nltk
import nltk.stem
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords

my_stem = nltk.stem.SnowballStemmer('english')

raw_data_file = "LDA/data/news.txt"
# filtered_data_file = './data/news_filtered.txt'
filtered_data_file = 'LDA/data/news_filtered.txt'

nltk.download('stopwords')
stopWords = set(stopwords.words('english'))


def reform_data():
    # stop_word_list = []
    # with open('./data/stopwords', 'r', encoding="UTF-8") as f:
    #     for line in f.readlines():
    #         stop_word_list.append(line.strip())

    res_lines = []
    with open(raw_data_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            temp = ""
            for character in line:
                if character == " " or character.isalpha():
                    temp += character
            filtered_line = ""
            for word in temp.lower().split():
                stemmed_word = my_stem.stem(word)
                if word not in stopWords:
                    filtered_line += stemmed_word + "\t"
            if not filtered_line == "":
                res_lines.append(filtered_line.strip())

    with open(filtered_data_file, 'w', encoding='UTF-8') as f:
        for line in res_lines:
            f.write("{}\n".format(line))


def my_lda_learn(topic):
    text = []
    with open(filtered_data_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            text.append(line.split())
    dictionary = Dictionary(text)

    text2bow = [dictionary.doc2bow(one_text) for one_text in text]

    my_lda = LdaModel(
        text2bow, id2word=dictionary, num_topics=topic, passes=20)

    print(my_lda.print_topics(num_topics=topic, num_words=10))


def lda_scratch(topic_num, alpha, beta, passes):
    docs_list = []
    with open(filtered_data_file, 'r', encoding="UTF-8") as f:
        for line in f.readlines():
            docs_list.append(line.split())
    dictionary = Dictionary(docs_list)

    docs_idx_list = []

    for doc in docs_list:
        one_doc_idx = []
        for word in doc:
            one_doc_idx.append(dictionary.token2id[word])
        docs_idx_list.append(one_doc_idx)

    doc_num = len(docs_list)
    word_num = len(dictionary.keys())

    n_d_k = np.zeros((doc_num, topic_num))
    n_k_w = np.zeros((topic_num, word_num))
    n_k = np.zeros((topic_num, ))

    z = {}

    for d, doc in enumerate(docs_idx_list):
        for w_index, w in enumerate(doc):
            k = np.random.randint(0, topic_num)
            n_d_k[d, k] += 1
            n_k_w[k, w] += 1
            n_k[k] += 1
            z[(d, w_index)] = k

    n_d_k = n_d_k + np.ones((topic_num, )) * alpha
    n_k_w = n_k_w + np.ones((word_num, )) * beta

    theta = np.zeros((doc_num, topic_num))

    for i_pass in range(passes):
        print("I_Pass: {}".format(i_pass))
        for d, doc in enumerate(docs_idx_list):
            theta[d] = np.random.dirichlet(
                n_d_k[d] + np.ones((topic_num, )) * alpha, 1)
            for w_index, w in enumerate(doc):
                word = w
                topic = z[(d, w_index)]

                n_d_k[d, topic] -= 1
                n_k_w[topic, word] -= 1
                n_k[topic] -= 1
                temp_phi = n_k_w[:, word] / n_k
                p_z_k = n_d_k[d] * temp_phi
                # p_z_k = theta[d] * temp_phi

                new_topic = np.random.multinomial(
                    1, p_z_k / np.sum(p_z_k)).argmax()

                z[(d, w_index)] = new_topic
                n_d_k[d, new_topic] += 1
                n_k_w[new_topic, word] += 1
                n_k[new_topic] += 1

    for k_i in range(topic_num):
        print("K: {}".format(k_i))
        arg_list = (n_k_w[k_i]).argsort()[-10:]
        for idx, arg_index in enumerate(list(reversed(arg_list))):
            print(
                "{}: {}".format(dictionary[arg_index],
                                n_k_w[k_i, arg_index] / np.sum(n_k_w[k_i])),
                end="\t")
        print("\n")


if __name__ == "__main__":
    # reform_data()
    my_lda_learn(5)
    # lda_sklearn(10)
    # lda_sklearn(20)
    # lda_scratch(5, 0.5, 0.5, 20)
    # lda_scratch(10, 0.5, 0.5,20)
    # lda_scratch(20, 0.5, 0.5,20)
