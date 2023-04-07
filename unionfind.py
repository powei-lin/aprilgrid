from dataclasses import dataclass

@dataclass
class Ufrec:
    # the parent of this node. If a node's parent is its own index,
    # then it is a root.
    parent: int # uint32_t 

    # for the root of a connected component, the number of components
    # connected to it. For intermediate values, it's not meaningful.
    size: int # uint32_t

class Unionfind:
    def __init__(self, _maxid: int) -> None:
        self.maxid = _maxid
        self.data = [Ufrec(i, 1) for i in range(_maxid)]
        

# this one seems to be every-so-slightly faster than the recursive
# version above.
def unionfind_get_representative(uf: Unionfind, id: int):
    root = id

    # chase down the root
    while (uf.data[root].parent != root):
        root = uf.data[root].parent

    # go back and collapse the tree.
    while (uf.data[id].parent != root):
        tmp = uf.data[id].parent
        uf.data[id].parent = root
        id = tmp

    return root

def unionfind_get_set_size(uf: Unionfind, id: int):
    repid = unionfind_get_representative(uf, id)
    return uf.data[repid].size

def unionfind_connect(uf: Unionfind, aid: int, bid: int):
    aroot = unionfind_get_representative(uf, aid)
    broot = unionfind_get_representative(uf, bid)

    if (aroot == broot):
        return aroot

    # we don't perform "union by rank", but we perform a similar
    # operation (but probably without the same asymptotic guarantee):
    # We join trees based on the number of *elements* (as opposed to
    # rank) contained within each tree. I.e., we use size as a proxy
    # for rank.  In my testing, it's often *faster* to use size than
    # rank, perhaps because the rank of the tree isn't that critical
    # if there are very few nodes in it.
    asize = uf.data[aroot].size
    bsize = uf.data[broot].size

    # optimization idea: We could shortcut some or all of the tree
    # that is grafted onto the other tree. Pro: those nodes were just
    # read and so are probably in cache. Con: it might end up being
    # wasted effort -- the tree might be grafted onto another tree in
    # a moment!
    if (asize > bsize):
        uf.data[broot].parent = aroot
        uf.data[aroot].size += bsize
        return aroot
    else:
        uf.data[aroot].parent = broot
        uf.data[broot].size += asize
        return broot
    
