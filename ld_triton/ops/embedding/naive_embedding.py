
import torch


class _naive_embedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
    ):  
        shape = weight.shape
        output = weight[input]
        ctx.save_for_backward(input)
        ctx.shape = shape
        return output
    
    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):
        input, = ctx.saved_tensors
        shape = ctx.shape
        grad_weight = torch.zeros(shape, device=grad_output.device)

        for i in range(shape[0]):
            grad_weight[i] += grad_output[input == i].sum(dim=0)
        return None, grad_weight

naive_embedding = _naive_embedding.apply


if __name__ == '__main__':
    factory_kwargs = {'device': 'cuda'}
    import torch.nn.functional as F
    _input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], **factory_kwargs)
    # _input = torch.tensor([[1, 1, 4, 5]], **factory_kwargs)
    _embedding_matrix = torch.rand(10, 3, requires_grad=True, **factory_kwargs)
    print(f'_embedding_matrix: {_embedding_matrix}')
    _output = F.embedding(_input, _embedding_matrix)
    d_output = torch.randn_like(_output)
    print(f'd_output: {d_output}')
    _output.backward(d_output)
    d_embedding_matrix, _embedding_matrix.grad = _embedding_matrix.grad.clone(), None

    naive_output = naive_embedding(_input, _embedding_matrix)
    naive_output.backward(d_output)
    naive_d_embedding_matrix, _embedding_matrix.grad = _embedding_matrix.grad.clone(), None
    assert torch.allclose(_output, naive_output)
    assert torch.allclose(d_embedding_matrix, naive_d_embedding_matrix), 'd_embedding_matrix: {}, naive_d_embedding_matrix: {}'.format(d_embedding_matrix, naive_d_embedding_matrix)
    
