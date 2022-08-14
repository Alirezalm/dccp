def gen_penalty(r, s, rho):
    miu = 10
    t = 2

    if r > miu * s:
        return t * rho
    elif s > miu * r:
        return rho / t

    return rho
