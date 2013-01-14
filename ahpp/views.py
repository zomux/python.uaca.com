from django.http import HttpResponse
from django.shortcuts import render_to_response
import StringIO


def dump_sense_tree(hptree,nodeId,level,stream=None):
    """
    @type hyp: GentileHypothesis
    """
    if level == 0 :
      prefix = ""
    else:
      prefix = "   "*(level-1)+"|--"
    tokens = []
    linkedNodes = []
    for tokenId in hptree.tree.node(nodeId):
      if tokenId > 0:
        word = hptree.tokens[tokenId-1][1]
        tokens.append(word)
      else:
        linkedNodeId = -tokenId
        linkedTokenId = hptree.mapNodeToMainToken[linkedNodeId]
        tag = hptree.tokens[linkedTokenId-1][0]
        tokens.append("[%s]" % (tag))
        linkedNodes.append(linkedNodeId)
    strTokens = " ".join(tokens)
    print >> stream, prefix + strTokens
    for linkedNodeId in linkedNodes:
      dump_sense_tree(hptree, linkedNodeId, level+1, stream)

def index(request):
  resultText = None
  if "action" in request.GET:
    if (request.GET["action"] == "parse" and "cfgtree" in request.POST
        and "deptree" in request.POST
        and request.POST["cfgtree"] and request.POST["deptree"]):
      pcfg = request.POST["cfgtree"]
      dep = request.POST["deptree"]
      from parser import sense
      t = sense.SenseTree(pcfg,dep)
      t.rebuildTopNode()
      t.appendXToTree()
      t.upMergeAllConjNodes()
      t.rebuildCommaNodes()
      t.separateContiniousNonTerminals()
      resultStream = StringIO.StringIO()
      dump_sense_tree(t, t.tree.root, 0, resultStream)
      resultText = resultStream.getvalue()

  return render_to_response('ahpp/templates/index.html', {"parse_result": resultText})