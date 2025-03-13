from syuron import shg


def use_chirped(L: float, Lambda0: float, kappa_val: float, chirp_rate: float = 0.0) -> shg.DomainStack:
    domains = []
    z = 0.0
    i = 0
    while z < L:
        current = Lambda0 / (1 + chirp_rate * z)
        width = round(current / 2, 2)
        kappa = kappa_val if i % 2 == 0 else -kappa_val
        domains.append(shg.Domain(width, kappa))
        z += current
        i += 1
    return domains
