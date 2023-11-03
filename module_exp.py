from typing import Sequence, Union

import cirq


class ModularExp(cirq.ArithmeticGate):
    """Квантовое модульное экспонирование.

    Этот класс вычисляет основание, возведенное в степень экспоненты, в заданном модуле: x^e mod n

        V|y⟩|e⟩ = |y * x^e mod n⟩ |e⟩      0 <= y < n
        V|y⟩|e⟩ = |y⟩ |e⟩                  n <= y

    где y - целевой регистр, e - регистр экспоненты, x - основание, n - модуль.
    Следовательно,

        V|y⟩|e⟩ = (U^e|r⟩)|e⟩

    где U - yнитарное преобразование, заданное как

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y

    Алгоритм квантового вычисления порядка (используется квантовое вычисление алгоритма Шора) использует
    квантовое модульное экспоненцирование вместе с квантовой оценкой фазы для вычисления порядка числа x
    по модулю n .
    """

    def __init__(
            self, target: Sequence[int], exponent: Union[int, Sequence[int]], base: int, modulus: int
    ) -> None:
        if len(target) < modulus.bit_length():
            raise ValueError(
                f'Регистр с {len(target)} кубитами слишком мал для модуля {modulus}'
            )
        self.target = target
        self.exponent = exponent
        self.base = base
        self.modulus = modulus

    def registers(self) -> Sequence[Union[int, Sequence[int]]]:
        return self.target, self.exponent, self.base, self.modulus

    def with_registers(self, *new_registers: Union[int, Sequence[int]]) -> 'ModularExp':
        if len(new_registers) != 4:
            raise ValueError(
                f'Ожидается 4 регистра (целевой, экспонента, основание и '
                f'модуль), но получено только {len(new_registers)}'
            )
        target, exponent, base, modulus = new_registers
        if not isinstance(target, Sequence):
            raise ValueError(f'Целевой должен быть регистром кубитов, а не {type(target)}')
        if not isinstance(base, int):
            raise ValueError(f'Основание должно быть константным числом, а не {type(base)}')
        if not isinstance(modulus, int):
            raise ValueError(f'Модуль должен быть константным числом, а не {type(modulus)}')
        return ModularExp(target, exponent, base, modulus)

    def apply(self, *register_values: int) -> int:
        assert len(register_values) == 4
        target, exponent, base, modulus = register_values
        if target >= modulus:
            return target
        return (target * base ** exponent) % modulus

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        assert args.known_qubits is not None
        wire_symbols = [f't{i}' for i in range(len(self.target))]
        e_str = str(self.exponent)
        if isinstance(self.exponent, Sequence):
            e_str = 'e'
            wire_symbols += [f'e{i}' for i in range(len(self.exponent))]
        wire_symbols[0] = f'ModularExp(t*{self.base}**{e_str} % {self.modulus})'
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))
