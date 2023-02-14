import torch

class buffer():
    def __init__(self):
        self.buffer_dict = dict()
    
    def update(self, key, value):
        self.buffer_dict.update({key:value})
    
    def clean(self, key='all'):
        if key == 'all':
            self.buffer_dict = dict()
        if key in self.buffer_dict:
            del self.buffer_dict[key]


class shape_vuer:
    def _shape_vue(self, x:list):
        if not isinstance(x[0], list):
            return [len(x), type(x[0])]
        
        shape = [len(x)] + self._shape_vue(x[0])
        return tuple(shape)

    def __call__(self, x:list):
        return self._shape_vue(x)

def shape_vue(x):
    vuer = shape_vuer()
    return vuer(x)

if __name__ == '__main__':
    test_ele = torch.Tensor([1,1,1])

    test_b = [[test_ele], [test_ele], [test_ele], [test_ele]]
    print(shape_vue(test_b))