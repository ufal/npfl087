#!/bin/python3
import pickle
import random
import jellyfish 

# Read file word by word
# if there exist substituion for this word
# then with given probability sample a substitute for it 
# (= sample from set of simillary sounding words given by the pickled dictionary)

class HomophoneNoise:
    def __init__(self, db_file, prob=0.3, threshold=0.8, top_k=3):
        self.prob = prob
        self.threshold = threshold
        self.top_k = top_k
        
        db = self.load_data(db_file)
        self.word_wID, self.wID_word, self.wID_setID_lst, self.setID_wID_lst = db
        
    def load_data(self, name):
        with open( name, 'rb') as f:
            return pickle.load(f)
        
    def noisify_word(self, word):
        # normlaize the word
        w1 = word.strip().strip(".").strip().strip(",").strip().strip("\"").strip().strip(",").strip()
        
        # TODO 
        # we are doing naive substituion
        # some representation words in the terms of features would be much better
        # two alternative improvements options:
        #   v1: create & store metaphone/soundex and compare with suitable algorithm  (match_rating_comparison)
        #   v2: create & store features which represents sounds... find in space using kNN / cos_sim ....
        if not w1 in self.word_wID:
            return  word
        
        # do the subsitution with given probability & threshodls
        else:
            # first obtain all similar words
            wID = self.word_wID[w1]
            lst_of_similar_words = []
            lst_setID = self.wID_setID_lst[wID] # word can possibly be part of multiple sets (homonyms are non transitive)
            for setID in lst_setID:
                wID_list = self.setID_wID_lst[setID]
                for wid in wID_list:
                    w = self.wID_word[wid]
                    lst_of_similar_words.append(w)
                    
            # next calculate some suitable score with all of them
            similar = []
            for w2 in lst_of_similar_words:
                if w1 == w2: # skip same [just a safety check ... will not happen]
                    continue
                elif jellyfish.match_rating_comparison(w1, w2): # must be phonetically similar
                        score = jellyfish.jaro_winkler(w1,w2)
                        if score > self.threshold:  # and score must be higher than threshold
                            similar.append((w2, score))

            if len(similar) == 0:
                return word
            
            # and sort them by this score
            similar = sorted(similar, key=lambda tup: tup[1], reverse=True)

            # sample those in range of threshols & top_k
            # ... todo
            
            # finally, with given probability
            if random.uniform(0, 1) >= 1-self.prob:
                idx = random.randint(0, len(similar)-1)
                w2, _ = similar[idx]
                print("returning w2")
                return w2
         
        # return the original word, if substituion was not returned ...
        print("returning original")
        return word
                
    
    def add_noise(self, src_file, target_file): # threshold can be >= 0.8 
        with open(src_file, 'r') as source:
            with open(target_file, 'w') as target:
                for line in source:
                    lst = line.strip().split()
                    target_line = ""
                    for i, word in enumerate(lst):
                        word = self.noisify_word(word)
                        if i > 0:
                            target_line += " "
                        target_line += word
                        
                    target_line += '\n'
                    target.write(target_line)

db_file = "honyme_db.pkl"

noiser = HomophoneNoise(db_file)

noiser.add_noise("./corpus", "./corpus.noise")
