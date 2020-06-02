#!/bin/bash

# Download cmudict
wget https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict

# Reformat to work with phonetisaurus
cat cmudict.dict | perl -pe 's/\([0-9]+\)//;
                             s/\s+/ /g; s/^\s+//;
                             s/\s+$//; @_ = split (/\s+/);
                             $w = shift (@_);
                             $_ = $w."\t".join (" ", @_)."\n";' > cmudict.formatted.dict