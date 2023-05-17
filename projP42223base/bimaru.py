# bimaru.py: Template para implementação do projeto de Inteligência Artificial 2022/2023.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from sys import stdin
import numpy as np

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class BimaruState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = BimaruState.state_id
        BimaruState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Bimaru."""
    
    """
    A representação vai ser feita através do Numpy
    Vamos colocar uma matriz com letras para representar o valor do ponto
    """
    def __init__(self, rows: list, collums: list):
        self.matrix = np.full((10,10), "0")
        self.row = rows
        self.collum = collums
        self.boatPositions  = ['C','T','M','B']
        """mais merdas ns oque fazer"""

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.matrix[row][col]

    def adjacent_vertical_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        num1 = None
        num2 = None

        if (0 < row-1 < 9):
            num1 = self.matrix[row - 1][col]
        
        if (0 < row + 1 < 9):
            num2 = self.matrix[row + 1][col]
        
        return (num1, num2)



    def adjacent_horizontal_values(self, row: int, col: int) -> (str, str):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        num1 = None
        num2 = None

        if (0 < col-1 < 9):
            num1 = self.matrix[row][col-1]
        
        if (0 < col + 1 < 9):
            num2 = self.matrix[row][col+1]
        
        return (num1, num2)
    
    def fill_water(self):
        """Verifica que partes tabuleiro ja podemos enher com agua"""
        
        for index in range(len(self.row)):

            if self.row[index] == 0:
                for i in range(10):
                    if (self.matrix[index][i] == "0"):
                        self.matrix[index][i] = "."
                self.row[index] = "-1"


            if self.collum[index] == 0:
                for i in range(10):
                    if (self.matrix[i][index] == "0"):
                        self.matrix[i][index] = "."
                self.collum[index] = "-1"
    
    def print_Board(self):
        """Imprime o tabuleiro no standard output."""
        board_string = ""
        for i in range(len(self.row)):
            for j in range(len(self.row)):
                board_string += self.matrix[i][j]
            board_string += "\n"
        print(board_string)
            

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 bimaru.py < input_T01

            > from sys import stdin
            > line = stdin.readline().split()
        """
        row = [int(x.strip()) for x in stdin.readline().split('\t')[1:]]
        print(row)
        collumn = [int(x.strip()) for x in stdin.readline().split('\t')[1:]]
        print(collumn)
        hints = int(stdin.readline())
        hints_list = []
        for i in range(hints):
            temp = stdin.readline().split('\t')
            hints_list.append([int(x.strip()) if x.strip().isdigit() else x.strip() for x in temp[1:]])
        print(hints_list)
        new_board = Board(row, collumn)
        for i in range(hints):
            new_board.matrix[hints_list[i][0]][hints_list[i][1]] = hints_list[i][2]
            if hints_list[i][2] in new_board.boatPositions:
                new_board.collum[hints_list[i][1]] -= 1
                new_board.row[hints_list[i][0]] -= 1
        
        return new_board



class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: BimaruState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        # TODO
        pass

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    ola = Board.parse_instance()
    ola.fill_water()
    ola.print_Board()
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
