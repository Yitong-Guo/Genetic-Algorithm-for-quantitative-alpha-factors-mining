import torch as t


def add_const(x, constant=1):
    """
    add constant value
    c = 1, 2, ..., 10
    """
    return x + constant


def mul_const(x, constant=1):
    """
    multiply a constant value
    c = 1, 2, ..., 10
    """
    return x * constant


def add(x, y):
    """
    plus
    """
    return x + y


def sub(x, y):
    """
    minus
    """
    return x - y


def mul(x, y):
    """
    multiply
    """
    return x * y


def div(x, y):
    """
    divide
    """
    return x / (y + 1e-6)


def log_torch(x):
    """
    log
    """
    return t.log(t.where(x > 0, x, t.tensor(1e-12)))


def exp_torch(x):
    """
    exp
    """
    return t.exp(x)


def sqrt_torch(x):
    """
    sqrt
    """
    return t.sqrt(t.where(x > 0, x, t.tensor(0.0)))


def square_torch(x):
    """
    square
    """
    return t.pow(x, 2)


def sin_torch(x):
    """
    sine
    """
    return t.sin(x)


def cos_torch(x):
    """
    cosine
    """
    return t.cos(x)


def neg(x):
    """
    opposite
    """
    return -x


def inv(x):
    """
    reciprocal
    """
    return 1 / (x + 1e-6)


def sign_torch(x):
    """
    sign
    """
    return t.sign(x)


def abs_torch(x):
    """
    abs
    """
    return t.abs(x)


def sigmoid_torch(x):
    """
    sigmoid
    """
    return t.sigmoid(x)


def haedsigmoid_torch(x):
    """
    hard sigmoid
    """
    return t.clamp((x + 1) / 2, min=0, max=1)


def leakyrelu_torch(x, alpha=0.1):
    """
    leaky relu
    alpha = 0, 0.1, ..., 0.9
    """
    return t.where(x > 0, x, x * alpha)


def gelu_torch(x):
    """
    gelu
    """
    return t.nn.functional.gelu(x)


if __name__ == '__main__':
    test_tensor = t.tensor([[1., 2., float('nan'), 4., 5.], [6., 7., float('nan'), 9., 10.]])
    y_tensor = t.tensor([[2., 2., 4., 4., 6.], [5., 5., 8., 9., 9.]])

    print(sigmoid_torch(test_tensor))
