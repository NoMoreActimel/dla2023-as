def get_padding(kernel_size, dilation):
    """ Returns padding for Conv1d, assuming stride = 1 """
    return dilation * (kernel_size - 1) // 2


def get_transposed_padding(kernel_size, stride):
    """ Returns padding for ConvTranspose1d, assuming dilation = 1 """
    return (kernel_size - stride) // 2 + stride % 2

def get_transposed_output_padding(kernel_size, stride):
    """ Returns output_padding for ConvTranspose1d, assuming dilation = 1 """
    return stride % 2

