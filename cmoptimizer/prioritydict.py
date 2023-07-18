import torch

class TensorList(dict):
    """List that can be used as a priority queue.

    The 'smallest' method can be used to return the object with lowest
    priority

    Reimplemented to avoid conflicts with other optimizers by setting hist=true.
    """

    def __init__(self, *args, **kwargs):
        super(TensorList, self).__init__(*args, **kwargs)
        self.aggr_sum = None
        # self.aggr_sq_sum = None
        self.smallest = 0

    def getNorms(self):
        return self._heap_key

    def size(self):
        return self.curr_k

    def setHyper(self, decay_rate=0.5, K=5, sampling='KotH', hist=False, dtype=None):
        self.k = K
        self.curr_k = 0
        self.decay_rate = decay_rate
        self.sampling = sampling
        self.hist = hist
        self.dtype = dtype


    def new_topc(self, new_k):
        if new_k<self.curr_k:
            if new_k==0:
                self.aggr_sum = self.aggr_sum*0.0
            else:
                _, indices = torch.topk(self._heap_key,new_k)
                self._heap_key = self._heap_key[indices]
                self._heap = self._heap[indices]
                self._heap_coeff = self._heap_coeff[indices]
                self.aggr_sum = torch.sum(self._heap, dim=0)
            self.k = new_k
            self.curr_k = new_k
            # print(self._heap_key[:self.curr_k], self._heap.shape)


    def addItem(self, key, val, alpha=1):
        if self.k==0:
            return
        if self.dtype is not None:
            val = val.to(dtype=self.dtype)
        if self.isFull():
            self._heap = self._heap.to(val.device)
            self.aggr_sum.add_(-self._heap[self.smallest])
            self._heap_key[self.smallest] = key
            self._heap[self.smallest] = val
        else:
            if self.curr_k==0:
                self._heap_key = torch.zeros(self.k, device=key.device, dtype=key.dtype)
                self._heap_coeff = torch.ones(self.k, device=key.device, dtype=key.dtype)
                self._heap = torch.zeros(self.k, *val.shape, device=val.device, dtype=val.dtype)
            self._heap_key[self.curr_k] = key
            self._heap[self.curr_k] = val
            self.curr_k += 1

        if self.aggr_sum is None:
            self.aggr_sum = torch.zeros_like(val)
            # self.aggr_sq_sum = torch.zeros_like(val)
        self.aggr_sum.add_(val, alpha=alpha)

    def pokeSmallest(self):
        """Return the lowest priority.
        Raises IndexError if the object is empty.
        """
        self.smallest = torch.argmin(self._heap_key)
        return self._heap_key[self.smallest]

    def isEmpty(self):
        return self.curr_k == 0

    def get_weighted_sum(self):
        weighted_sum = (1-self._heap_coeff[0])*self._heap[0]
        for j in range(1,self.curr_k):
            weighted_sum = weighted_sum.add_(self._heap[j], alpha=1-self._heap_coeff[j])
        return weighted_sum

    def decay_vals(self, factor):
        self._heap = torch.mul(self._heap, factor)

    def decay_coeff(self, factor):
        self._heap_coeff = torch.mul(self._heap_coeff, factor)

    def decay(self):
        self._heap_key = torch.mul(self._heap_key, self.decay_rate)

    def updateKey(self, val, key):
        index = (self._heap == val).nonzero(as_tuple=True)
        if len(index[0])>0:
            self._heap_key[int(index[0][0])] = key

    def isFull(self):
        return self.curr_k == self.k # len(self._heap) >= self.k

    def averageTopC(self):
        average_topc = 0.
        if self.curr_k > 0:
            if not self.hist:
                average_topc = torch.sum([it.norm() for it in self._heap]) / float(self.curr_k)
            else:
                average_topc = torch.sum([it.g.norm() for it in self._heap]) / float(self.curr_k)
        return average_topc

    def getMin(self):
        """
        Get smallest gradient
        :return: The smallest gradient
        """
        return self._heap[self.smallest]

    def getMax(self):
        "Returns the largest gradient"
        return self._heap[torch.argmax(self._heap_key)]

    def __getitem__(self, key):
        return self._heap[self._heap_key==key]

    def __len__(self):
        return self.curr_k

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def step(self):
        for item in self._heap: item.step()

    def epoch(self):
        ages = []
        for item in self._heap:
            ages.append(item.epoch_age)
            item.resetEpoch()
        return ages
