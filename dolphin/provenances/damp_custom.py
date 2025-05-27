import torch

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if len(a.shape) == 1:
        a = a.view(1, -1)

    if len(b.shape) == 1:
        b = b.view(1, -1)

    # print(a.shape, b.shape)
    
    assert len(a.shape) == 2
    batch_size = a.shape[0]
    num_classes_a = a.shape[-1]
    num_classes_b = b.shape[-1]

    num_res = num_classes_a + num_classes_b - 1

    res = torch.zeros(batch_size, num_res)

    halfway = (num_res) // 2

    tensor_a = a.view(batch_size, -1, 1)
    tensor_b = b.view(batch_size, 1, -1)

    # product = torch.t(tensor_a) * tensor_b
    product = tensor_a @ tensor_b
    # product = torch.fliplr(product)
    product = torch.flip(product, dims=(-1,))

    for i in range(num_res):
        offset = halfway - i
        res[:, i] = torch.diagonal(product, dim1=-2, dim2=-1, offset=offset).sum(dim=-1)

    return res.clamp(min=0.0, max=1.0)
