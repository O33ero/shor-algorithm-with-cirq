import fractions
import math
import random
import time
from typing import Callable, Optional

import cirq
import sympy

from module_exp import ModularExp


def naive_order_finder(x: int, n: int) -> Optional[int]:
    """Нахождение наименьшего числа r, для которого x^r mod n == 1ю

    Метод: "в тупую" через последовательный перебор

    Арг.:
        x: число для, которого вычисляется порядок. Должно быть >1 и <n,
        n: модуль мультипликативной группы

    Возвращает:
        Наименьшее положительное число r, для которого x^r mod n == 1.
    """
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')
    r, y = 1, x
    while y != 1:
        y = (x * y) % n
        r += 1
    return r


def quantum_order_finder(x: int, n: int) -> Optional[int]:
    """Нахождение наименьшего числа r, для которого x^r mod n == 1

    Метод: через квантовый алгоритм Шора

    Арг.:
        x: число для, которого вычисляется порядок. Должно быть >1 и <n,
        n: модуль мультипликативной группы.

    Возвращает:
        Наименьшее положительное число r, для которого x^r mod n == 1.
    """
    # Проверка что число x является элементов мультипликативной группы по модулю n
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')

    # Создание схемы вычисления порядка.
    circuit = make_order_finding_circuit(x, n)

    # Дискретизация схемы вычисления порядка.
    measurement = cirq.sample(circuit)

    # Результат вычисления выходного значения.
    return process_measurement(measurement, x, n)


def make_order_finding_circuit(x: int, n: int) -> cirq.Circuit:
    """Возвращает квантовую схему, для вычисления порядка x по модулю n .

    Схема использует оценку квантовой фазы, для вычисление собственного
    унарного значения

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y

    Схема использует 2 регистра: регистр, на который воздействует U, и
    регистр экспоненты, из которого после измерения высчитывается собственное
    значение. Cхема состоит из 3-х этапов:

    1. Инициальзиация регистра в состояние |0..01⟩ и регистра экспоненты в
       состояние суперпозиции.
    2. Реализация оператора умножения U^2^j с помощью модулярной экспоненты.
    3. Обратное квантовое преобразование Фурье для переноса собственного 
       значения в регистр экспоненты.

    Арг.:
        x: положительное число, для окторого вычисляется порядок по модулю n
        n: модуль, относительно которого вычисляется порядок числа x

    Возвращает:
        Квантовую схему для вычисления порядка x по модулю n
    """
    L = n.bit_length()
    target = cirq.LineQubit.range(L)
    exponent = cirq.LineQubit.range(L, 3 * L + 3)
    return cirq.Circuit(
        cirq.X(target[L - 1]),
        cirq.H.on_each(*exponent),
        ModularExp([2] * len(target), [2] * len(exponent), x, n).on(*target + exponent),
        cirq.qft(*exponent, inverse=True),
        cirq.measure(*exponent, key='exponent'),
    )

def process_measurement(result: cirq.Result, x: int, n: int) -> Optional[int]:
    """Вычисление выходного значения из квантовой схемы вычисления порядка.

    В частности, определеняет s/r такое, что exp(2πis/r) является собственным
    унарным значением

        U|y⟩ = |xy mod n⟩  0 <= y < n
        U|y⟩ = |y⟩         n <= y

    после этого вычисляется r, если возможно, и возвращает его.

    Арг.:
        result: результат, полученный при дискретизации схемы построенной с 
        помощью make_order_finding_circuit

    Возвращает:
        r, порядок числа x по модулю n или None.
    """
    # Вычисление выходящего значения из регистра экспоненты.
    exponent_as_integer = result.data["exponent"][0]
    exponent_num_bits = result.measurements["exponent"].shape[1]
    eigenphase = float(exponent_as_integer / 2 ** exponent_num_bits)

    # Вычисление f = s / r.
    f = fractions.Fraction.from_float(eigenphase).limit_denominator(n)

    # Если нумератор = 0, результат не получен.
    if f.numerator == 0:
        return None

    # Иначе, возвращается знаменатель.
    r = f.denominator
    if x ** r % n != 1:
        return None
    return r


def find_factor_of_prime_power(n: int) -> Optional[int]:
    """Возвращает нетривиальный коэффициент факторизации для n, если n степень простого числа, иначе None."""
    for k in range(2, math.floor(math.log2(n)) + 1):
        c = math.pow(n, 1 / k)
        c1 = math.floor(c)
        if c1 ** k == n:
            return c1
        c2 = math.ceil(c)
        if c2 ** k == n:
            return c2
    return None


def find_factor(
        n: int,
        order_finder: Callable[[int, int], Optional[int]] = quantum_order_finder,
        max_attempts: int = 5
) -> Optional[int]:
    """Возвращает нетривиальный коэффициент факторизации для составного положительного числа n.

    Арг.:
        n: число для факторизации.
        order_finder: Функция для нахождения порядка элементов мультипликативной группы чисел по модулю n.
        max_attempts: количество попыток для вычисления порядка с помощью order_finder.

    Возвращает:
        Нетривиальный коэффциент факторизации числа n или None, если не удалось найти  .
        Коэффициент факторизации k называют тривилальный, если k равен 1 или n.
    """
    if order_finder == naive_order_finder:
        print("Классический метод факторизации: ")
    if order_finder == quantum_order_finder:
        print("Квантовый метод факторизации: ")

    # Если число n простое, то у него нет нетривиального коэффициента.
    if sympy.isprime(n):
        print("n простое!")
        return None

    # Если число n четное, то у него 2 нетривильных коэффициента.
    if n % 2 == 0:
        return 2

    # Если число n степень простого числа, то можно вычислить коэффициент проще.
    c = find_factor_of_prime_power(n)
    if c is not None:
        return c

    for _ in range(max_attempts):
        # Выбираем любое число принадлежащее [2...n - 1].
        x = random.randint(2, n - 1)

        # Скорее всего x и n будут взаимно простыми.
        c = math.gcd(x, n)

        # Если x и n не взаимно простые, то нам повезло - мы наши
        # нетривиальный коэффициент.
        if 1 < c < n:
            return c

        # Вычисление порядка r для числа x по модулю n используя функцию вычисления порядка.
        r = order_finder(x, n)

        # Если вычисление порядка неудачно, попробуем еще раз.
        if r is None:
            continue

        # Если порядок четный, попробуем еще раз.
        if r % 2 != 0:
            continue

        # Вычисление нетривиального коэффициента.
        y = x ** (r // 2) % n
        assert 1 < y < n
        c = math.gcd(y - 1, n)
        if 1 < c < n:
            return c

    print(f"Не удалось найти нетривиальный коэффициент факторизации за {max_attempts} попыток.")
    return None


if __name__ == '__main__':
    n = 3 * 11
    start = time.time()
    p = find_factor(n, order_finder=quantum_order_finder)
    q = n // p
    end = time.time()

    print("Факторизация n = pq =", n)
    print("p =", p)
    print("q =", q)
    print("Результат pq равен n?", p * q == n)
    print(f"Выполнено за {end - start:.5f} секунд(ы)")
