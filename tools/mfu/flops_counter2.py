def llm_flops(seq_list):
    h = 1536
    intermediate_size = 8960
    attention = 0
    seq_sum = 0
    for s in seq_list:
        attention += 2 * h * s * s
        seq_sum += s
    return (8 * seq_sum * h * h + attention + 6 * seq_sum * h * intermediate_size)  * 28 * 3 / 1e12


def vit_flops(image_list):
    h = 1024
    attention = 0
    seq_sum = 0
    for s in image_list:
        attention += 4 * s * h * 1024 * 1024
        seq_sum += s * 1024
    return (24 * seq_sum * h * h + attention) * 24 * 3 / 1e12



