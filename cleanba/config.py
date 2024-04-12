import random


def random_seed() -> int:
    return random.randint(0, 2**31 - 2)
