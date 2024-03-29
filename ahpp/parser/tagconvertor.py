# Simplified
TAGMAP={
  "NN" : "NP",
  "JJ" : "JJ",
  "CD" : "CD",
  "," : "NP",
  "DT" : "DT",
  "NNS" : "NP",
  "CC" : "NP",
  "IN" : "IN",
  "VBG" : "VP",
  "VBN" : "VP",
  "." : "S",
  "VBZ" : "VP",
  "''" : "NP",
  "RB" : "NP",
  "NNP" : "NP",
  "-RRB-" : "NP",
  "SYM" : "NP",
  "TO" : "TO",
  "VB" : "VP",
  "FW" : "NP",
  "-LRB-" : "NP",
  "PRP" : "NP",
  "VBP" : "VP",
  "JJR" : "NP",
  ":" : "NP",
  "WRB" : "NP",
  "MD" : "NP",
  "WDT"  : "NP",
  "VBD" : "VP",
  "EX" : "NP",
  "RBR" : "NP",
  "PRP$" : "NP",
  "#" : "NP",
  "PDT" : "NP",
  "JJS" : "NP",
  "LS" : "NP",
  "POS" : "NP",
  "UH" : "NP",
  "RP" : "NP",
  "$" : "NP",
  "WP$" : "NP",
  "``" : "NP",
  "RBS" : "NP",
  "WP" : "NP",
  "NNPS" : "NP",
}
# Not Simplified
TAGMAP={
  "NN" : "NP",
  "JJ" : "JJ",
  "CD" : "CD",
  "," : "S",
  "DT" : "DT",
  "NNS" : "NP",
  "CC" : "S",
  "IN" : "IN",
  "VBG" : "VP",
  "VBN" : "VP",
  "." : "S",
  "VBZ" : "VBZ",
  "''" : "NP",
  "RB" : "X",
  "NNP" : "NP",
  "-RRB-" : "X",
  "SYM" : "X",
  "TO" : "TO",
  "VB" : "VBZ",
  "FW" : "X",
  "-LRB-" : "X",
  "PRP" : "X",
  "VBP" : "VBZ",
  "JJR" : "NP",
  ":" : "X",
  "WRB" : "S",
  "MD" : "X",
  "WDT"  : "X",
  "VBD" : "VP",
  "EX" : "X",
  "RBR" : "X",
  "PRP$" : "X",
  "#" : "X",
  "PDT" : "X",
  "JJS" : "X",
  "LS" : "X",
  "POS" : "X",
  "UH" : "X",
  "RP" : "X",
  "$" : "X",
  "WP$" : "X",
  "``" : "X",
  "RBS" : "X",
  "WP" : "X",
  "NNPS" : "NP",
}

def convert_tags_for_tokens(tokens):
  """
  Convert for tokens.
  """
  newTokens = []
  for token in tokens:
    tag, word = token
    if tag in TAGMAP:
      tag = TAGMAP[tag]
    newTokens.append((tag, word))
  return newTokens