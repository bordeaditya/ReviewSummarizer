#######################################################################################
# Author  : Aditya Borde
# UTD Id  : asb140930
# Project : Movie Review Summarizer
#######################################################################################
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import defaultdict as ddict
from string import punctuation
import string
import re
from heapq import nlargest
import urllib2
import sys
import networkx as nx
import itertools
import time
from bs4 import BeautifulSoup
from math import log10

#######################################################################################
##################         Movie Review Summarizer  III      ##########################
#######################################################################################
class TextRankingReviewSummarizer(object):
    ################## Constructor #####################################################
    def __init__(self,title):
        self._title = title
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    ################### Lin Similarity ################################################
    def _get_edge_weight(self,first,second):
        sent_words_1 = first.split()
        sent_words_2 = second.split()
        common_count = len(set(sent_words_1) & set(sent_words_2))
        normalization_factor = log10(len(sent_words_1)) + log10(len(sent_words_2))
        if normalization_factor == 0: return 0
        else: return common_count / normalization_factor

    ################## Building graph ###############################################
    def _build_graph(self,g_nodes):
        graph = nx.Graph()
        graph.add_nodes_from(g_nodes)
        #### create all combinations of sent_tokens - order is not important
        pair_nodes = list(itertools.combinations(g_nodes,2))
        ###### create edge between links with weighted similarity
        for pair_node in pair_nodes:
            first_sent = pair_node[0]
            second_sent = pair_node[1]
            node_weights = self._get_edge_weight(first_sent,second_sent)
            graph.add_edge(first_sent, second_sent, weight=node_weights)
        return graph

    ####### sorting sentences ###############
    def _get_by_ranks(self,ranks,n):
        return nlargest(n, ranks, key=ranks.get)

    ####### final Summary ###################
    def moview_summary(self,text,n):
        sent_tokens = sent_tokenize(text.strip())
        graph = self._build_graph(sent_tokens)

        rank = nx.pagerank(graph,weight='weight')
        sents_by_rank = self._get_by_ranks(rank,n)

        return sents_by_rank


#######################################################################################
##################         Movie Review Summarizer II        ##########################
#######################################################################################
class TitleBasedReviewSummarizer(object):
    ############ Constructor to Set the initial Values #################################
    def __init__(self,title):
        self._title = title
        self._stopwords = set(stopwords.words('english') + list(punctuation))
        self._list_synsets = []
        self._list_original = []
        self._dict_signs = {}

    ############# form a required dictioanry  ########################################
    def _form_dictionary_signs(self,tokens_title):
        for word in tokens_title:
            word_synsets = wordnet.synsets(word.strip())
            self._list_synsets.append(wordnet.synsets(word.strip()))
            if word_synsets:
                self._list_original.append(word_synsets[0])
            else:
                self._list_original.append(None)

            for element in word_synsets:
                sign = set()
                list_temp = [element.definition()], element.examples()
                for temp in list_temp:
                    for item in temp:
                        item = item.encode('ascii')
                        actual_word = item.translate(None,string.punctuation)
                        sign.update(actual_word.lower().split())
                self._dict_signs[element] = sign - self._stopwords

    ############# Identify  Best meaning in title #####################################
    def _get_synsets(self,movie_title):
        ########## create required dictioary first ########
        movie_title = map(lambda e:e.encode('ascii'),movie_title)
        self._list_synsets = filter(None,self._list_synsets)
        print
        self._form_dictionary_signs(movie_title)
        for index, list_synset in enumerate(self._list_synsets):
            list_current = []
            list_current.extend(self._list_synsets[:index])
            list_current.extend(self._list_synsets[index + 1:])
            #### Get the sets
            set_current_sign = set()

            for current in list_current:
                for item_synset in current:
                    set_current_sign.update(self._dict_signs[item_synset])

            dict_overlaps = {}
            for item in list_synset:
                set_overlaps = self._dict_signs[item] & set_current_sign
                dict_overlaps[item] = len(set_overlaps)

            if list_synset:
                max_set = max(list_synset,key = dict_overlaps.get)
                current_count = dict_overlaps[max_set]
                if current_count == 0:
                    max_set = self._list_original[index]
                self._list_synsets[index] = [max_set]

        return [synsets[0] if synsets else None for synsets in self._list_synsets]

    ########################### find similarity with Sentence ########################
    def movie_title_review_summary(self,review,n):
        title = self._title
        clean_title = re.sub('[^A-Za-z\.]+', ' ', self._title)
        clean_title = clean_title.strip()
        tokens_title = word_tokenize(clean_title)
        ressets = self._get_synsets(tokens_title)
        synsets  = [x for x in ressets if x is not None]
        list_signs = [synset.definition() for synset in synsets]

        list_examples= [synset.examples() for synset in synsets]
        list_examples = sum(list_examples,[])
        signatures = list_examples + list_signs
        signatures = filter(None, signatures)

        signatures = [word_tokenize(w) for w in signatures]
        ## Flatten all list of lists to set
        observations = set(sum(signatures,[]))
        actual_observations = set(tokens_title)
        observations = observations.union(actual_observations)
        observations = set(map(lambda e: e.lower(),observations))
        self._relevent_wordset = observations - self._stopwords
        sentence_tokens = sent_tokenize(review)
        ranks = ddict(int)
        ######### Calculate Frequency ###########################
        for i,sent in enumerate(sentence_tokens):
            formation_words = set(word_tokenize(sent))
            formation_words = set(map(lambda e: e.lower(),formation_words))
            similarity_value = len(self._relevent_wordset & formation_words)
            if ':' not in formation_words:
                ranks[i] += similarity_value
            else:
                ranks[i] = -sys.maxint

        sentence_indices = self._get_best_tokens_wrt_title(n,ranks)
        sentence_indices.sort()
        return [sentence_tokens[j] for j in sentence_indices]

    def _get_best_tokens_wrt_title(self, n, dict_ranks):
        return nlargest(n, dict_ranks, key=dict_ranks.get)



#######################################################################################
##################         Movie Review Summarizer  I        ##########################
#######################################################################################
class NaiveBasedReviewSummarizer(object):
    ############ Constructor to Set the initial Values #################################
    def __init__(self, min_threshold, max_threshold,title):
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._title = title
        self._stopwords = set(stopwords.words('english'))

    ############ Calculating Word Frequencies #########################################
    def _calculate_word_frequencies(self, word_tokens):
        freq = ddict(int)
        for s in word_tokens:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1

        normalization_factor = float(max(freq.values()))
        for w in freq.keys():
            ####### Normalizing Frequency ########################
            freq[w] = freq[w] / normalization_factor
            if freq[w] >= self._max_threshold or freq[w] <= self._min_threshold:
                del freq[w]
        return freq

    ########## Moview Summary ########################################################
    def movie_summary(self, review, n):
        sentence_tokens = sent_tokenize(review)
        word_tokens = [word_tokenize(s.lower()) for s in sentence_tokens]
        self._freq = self._calculate_word_frequencies(word_tokens)
        ranks = ddict(int)
        list_index = list(xrange(len(sentence_tokens)))

        for i, sent in zip(list_index,word_tokens):
            for w in sent:
                if w in self._freq:
                    if ':' not in w:
                        ranks[i] += self._freq[w]
                    else:
                        ranks[i] -= self._freq[w]

        sentence_indices = self._get_best_tokens(n,ranks)
        sentence_indices.sort()
        return [sentence_tokens[j] for j in sentence_indices]

    ########## Get the Best n - Sentence Tokens that has maximum frequency Value ###########
    def _get_best_tokens(self, n, dict_ranks):
        return nlargest(n, dict_ranks, key=dict_ranks.get)

#####################################################################################
#####################       Parse Input Data         ################################
#####################################################################################
def parse_input_data(link):
    data = urllib2.urlopen(input_url).read().decode('utf8')
    soup = BeautifulSoup(data,'html.parser')
    movie_name = soup.find('title').get_text()
    moview_review = ' '.join(map(lambda para: para.text, soup.find_all('p')))
    moview_review = moview_review.replace('\n\r',' ')
    return movie_name, moview_review

#####################################################################################
#####################       Display review Summary    ###############################
#####################################################################################
def display_summary(title,summary,type):
    print '-----------------------------------------------------------------'
    print type + ' : ' + title
    print '-----------------------------------------------------------------'
    review = ''.join(map(lambda s: s.strip(),summary))
    print review
    print '-----------------------------------------------------------------'


######### Execution starts from Here: ###############################################
if __name__ == "__main__":
    input_url = raw_input('Enter local url to get the movie review :')
    input_size = raw_input('Enter the number by which you want to reduce the Review [e.g: 3 means reduces to (1/3)rd]:')
    size = int(input_size)
    # input_url = 'http://localhost/MovieReviews/0016.html'
    # size = 5
    min_threshold = 0.2
    max_threshold = 0.8
    movie_name, movie_review = parse_input_data(input_url)
    reduction_size = int(round(len(sent_tokenize(movie_review))/size))

    ################### Naive Review Summary Approch ###################################
    start_naive = time.clock()
    obj_movie_review = NaiveBasedReviewSummarizer(min_threshold,max_threshold,movie_name)
    movie_summary = obj_movie_review.movie_summary(movie_review,reduction_size)
    display_summary(movie_name,movie_summary,'Naive Based Summary')
    end_naive = time.clock()

    ################### Title Based Summary Approch ###################################
    start_title = time.clock()
    movie_title = movie_name.replace('Review for','').strip()
    obj_movie_name_review = TitleBasedReviewSummarizer(movie_title)
    summary_wrt_title = obj_movie_name_review.movie_title_review_summary(movie_review,reduction_size)
    display_summary(movie_name,summary_wrt_title,'Title Based Summary')
    end_title = time.clock()

    ################### Text Rank Based Summary Approch #################################
    start_text = time.clock()
    obj_textrank = TextRankingReviewSummarizer(movie_name)
    summary_text_rank = obj_textrank.moview_summary(movie_review,reduction_size)
    display_summary(movie_name,summary_text_rank,'Text Rank Based Summary')
    end_text = time.clock()

    ################### PREFORMANCE #####################################################
    print '------------------Evaluation-------------------------------------'
    print 'Naive Based Summary\t\t= {0:.2f} seconds'.format(end_naive-start_naive)
    print 'Title Based Summary\t\t= {0:.2f} seconds'.format(end_title-start_title)
    print 'Rank  Based Summary\t\t= {0:.2f} seconds'.format(end_text-start_text)
    print '------------------Evaluation-------------------------------------'
    print 'END'