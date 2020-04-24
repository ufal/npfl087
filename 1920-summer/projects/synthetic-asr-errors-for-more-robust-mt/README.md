# Generating ASR errors using the homonyme / homophone database

## Info about the files

* database was created out of three inputs:
    * from homonymes scraped from the web
    * from homonymes created by aligning asr_output with human_output using GIZA++
    * from aligning common english words (370k) using combination of match_rating_comparison algorithm with jaro_winkler
        * this last input is still running and will be finished on sunday
        
* there are some jupyter notebooks that were used to create the database. Namely:
    * `homonyms-giza.ipynb`  ... for parsing GIZA++ output
    * `homonyms-scrape.ipynb` ... for parsing scraped homonyms
    * `homophonyms-from-english-words.ipynb` ... for aligning words from english dictionary
    
    * `pickle-the-dataset.ipynb` ... for creating the final pickled database from multiple inputs
    
* raw data from which the database was created are in these folders:
    * `./homonyms-scraped/` ... contains web-scraped homonyms
    * `./english-words/` ... containd english common words dictionaries
    * `./homonyms-data/` ... contains parsed txt files from which final database pickle is crated
    
## Usage
* in order to add noise to some text you can use HomophoneNoise class in `add-homonym-noise.ipynb`
* constructor takes following arguments:
    * `db_file` ... pickled database ... right now there is `honyme_db.pkl` available
    * `prob` ...  if the word can be noisified (e.g. there is substitute in DB), substitute with this probability
    * `threshold` ... even though database contains sets of similar words, threshold can be additionaly used to limit the similar words in the terms of jaro_winkler similaryty (use something beteween 0.8 and 1.0) 1.0 .. all available similar words are considered for substituion
    * `top_k` ... additional parameter for sampling the substitute word only from top_k scoring similar word ( in the terms of jaro_winkler probability)
    
    
    
 ## Sample output
 * `./human_out.txt` was processed and the noisified output is in the `./human_out.txt.noise` file
 
 
 ## Summary
 * It works quite good, the most common ASR errors - homonymes / homophones are targeted
 * Database can be easily extended using other alignments from GIZA++ or word alignments ...
 
 ## Other considerations
 * We observed, that the second most common type of ASR errors are "senteces" - ASR tends to create shorter sentences
     * Some sentence splitting of the can be used in order to create this type of noise
     
  
 * the "database of words" approach is not ideal and some features would be more suitable
 * we found few paper that describes how to create features from words, where similar sounding words are very close in feature space
 * we want to further investigate this approach and use it instead of this "word" approach