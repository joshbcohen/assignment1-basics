import numpy.typing as npt
import numpy as np
import random

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):

    max_idx = min(len(dataset) - context_length -1, batch_size*context_length)
    starts = [random.randint(0, max_idx + 1) for _ in range(batch_size)]
    o1 = np.array([dataset[s:s+context_length] for s in starts])
    o2 = np.array([dataset[s+1:s+1+context_length] for s in starts])
    
    return (o1, o2)


    # output dim should be batch_size, context_length

if __name__ == "__main__":
    x = np.arange(100)
    a, b = data_loading(dataset=x, batch_size=5, context_length=10, device = "cpu")
    print(a)




