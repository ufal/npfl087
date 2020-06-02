# Dictionaries
* HomoNoise support two types of pickled dictionaries: **Hash Tables** and **Dictionaries**
* You can provide arbitrary number of dictionaries to the HomoNoiser
* All words in dictionaries are expected to be lowercased 

### 1. Hash Tables
 * format: `hash_dict = { h: [s1, s2, ... ] } ` 
 * where:
    * `h` -  phonetic hash  (soundex, metaphone ...)
    * `s1,s2, ...` - strings that are hashed to phonetic hash `h`
    * when you provide the hash table, you must specify:
       * what hash `h` is used
       * what mapping does the hash table
          * `1:1` - `h` is created from 1 word `si` is crated form 1 word
          * `1:2` -  `h` is from 1 word  `si = "w1 w2"` for words w1, w2
          * etc ...
      * format is specified by the naming convention of the provided pickled file
        * `{your-name}_{HASH}_{M}_{N}.pkl`
        * example: hash_table using `1:2` mapping and METAPHONE hash must be named `foo-bar_METAPHONE_1_2.pkl`

##### HASH TABLE DATA PREPARATION
 * there are few hash tables provided in the `hashtable/` folder
 * New hash table can be created using ``dictionary_sound_hash.py`` script
 * you need to provde list of words (like ``data/words-400K`` or ``data/words-30K``) 
 * 1:1 mapping creates relatively small file ~ 2*size_of_list
   * note: that this can be used for M:1 error search also for `h = hash("w1w2")` 
 * 1:2 mapping can create huge files ...  (1000*size_of_list)
   * so it is not worth it for large word lists
   * note: again, can be used for M:2 error search
  * larger mapping is not possible due to exponential grow of file size 


### 2. Dictionaries
* format: `dict = { s: [s1, s2 ... ] }`
* where:
    * `s` is word
    * `si` is 'similar sounding word'; we expect that s != si 

#####  DICTIONARY DATA PREPARATION
* there are few dictionaries provided in the `dictionary/` folder
* New dictionary can be created using:
  1. ``dictionary_from_alignment.py`` script
     * provide list of words (like ``data/words-400K`` or ``data/words-30K``) 
     * words all considered similar by Match Rating Algorithm
     * Because this comparison is too loose (lot of words are similar),
      we use Jaro-Winkler similarity to bound  this similarity which returns for s1,s2 returns number from [0, 1]
      where 
        * 0 ...completly dissimilar
        * 1 ... same
      *  ``dictionary_from_alignment.py`` have `threshold` parameter for the Jaro-Winkler similarity
      * usually `threshold=0.8` provides words that are reasonably similar (also there isn't some insane number of words considered similar and file size won't explode)
  2. using some other script that provides dictionary in specified format
    * (for instance using the `dictionary_from_scrape.py`  which processes scraped homophones in `data/homonyms-scraped/`)

