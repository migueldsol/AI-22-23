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
        self.matrix = np.full((10, 10), "0")
        self.row = rows
        self.collum = collums
        self.boatPositions = ["C", "T", "M", "B", "L", "R"]
        self.BoatSizes = {4: 1, 3: 2, 2: 3, 1: 4}   
        self.boatParts = {"M": 4, "O": 12}          #tamos a contar os Middles e os Top/Bot/Right/Left para verificações
                                                    #já contamos os Center no BoatSizes

    def get_value(self, row: int, col: int) -> str:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.matrix[row, col]

    def highest_boat_size(self):
        return max(self.BoatSizes.keys())

    def remove_boat_possibility(self, size):
        self.BoatSizes[size] -= 1

    def check_boat_position(self,position:str):
        return True if position in self.boatPositions else False
        
    def adjacent_vertical_values(self, row: int, col: int) -> tuple((str, str)):
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""
        num1 = None
        num2 = None

        if 0 < row - 1 < 9:
            num1 = self.matrix[row - 1, col]

        if 0 < row + 1 < 9:
            num2 = self.matrix[row + 1, col]

        return (num1, num2)

    def adjacent_horizontal_values(self, row: int, col: int) -> tuple((str, str)):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        num1 = None
        num2 = None

        if 0 < col - 1 < 9:
            num1 = self.matrix[row, col - 1]

        if 0 < col + 1 < 9:
            num2 = self.matrix[row, col + 1]

        return (num1, num2)

    def fill_water(self):
        """Verifica que partes tabuleiro ja podemos encher com agua"""

        for index in range(len(self.row)):          #Percorre as rows e collums a procura de 0 partes de barco
            if self.row[index] == 0:                #Se encontrar valor por colocar coloca "."
                for i in range(10):
                    if self.matrix[index, i] == "0":
                        self.matrix[index, i] = "."
                self.row[index] = -1              #Coloca a row a -1 para indicar que já foi tratada

            if self.collum[index] == 0:            #Se encontrar valor por colocar coloca "."
                for i in range(10):
                    if self.matrix[i, index] == "0":
                        self.matrix[i, index] = "."
                self.collum[index] = -1           #Coloca a collum a -1 para indicar que já foi tratada
                
    def piece_water_spaces(self, row: int, column: int, letter: str):
        # do a switch case for each letter fill the surrounding with water(T -> top,left,right, B, Middle->M)
        if letter.capitalize() == "T":
            if row > 0:
                self.matrix[row - 1, column] = "."
            if column > 0:
                self.matrix[row, column - 1] = "."
            if column < 9:
                self.matrix[row, column + 1] = "."
            if column > 0 and row > 0:
                self.matrix[row - 1, column - 1] = "."
            if column < 9 and row > 0:
                self.matrix[row - 1, column + 1] = "."
        elif letter.capitalize() == "B":
            if row < 9:
                self.matrix[row + 1, column] = "."
            if column > 0:
                self.matrix[row, column - 1] = "."
            if column < 9:
                self.matrix[row, column + 1] = "."
            if column > 0 and row < 9:
                self.matrix[row + 1, column - 1] = "."
            if column < 9 and row < 9:
                self.matrix[row + 1, column + 1] = "."
        # middle only left and right positions are filled with water
        elif letter.capitalize() == "M":
            if column > 0 and row > 0:
                self.matrix[row - 1, column - 1] = "."
            if column < 9 and row > 0:
                self.matrix[row - 1, column + 1] = "."
            if column > 0 and row < 9:
                self.matrix[row + 1, column - 1] = "."
            if column < 9 and row < 9:
                self.matrix[row + 1, column + 1] = "."
        # c - all around is filled with water
        elif letter.capitalize() == "C":
            if row > 0:
                self.matrix[row - 1, column] = "."
            if column > 0:
                self.matrix[row, column - 1] = "."
            if column < 9:
                self.matrix[row, column + 1] = "."
            if row < 9:
                self.matrix[row + 1, column] = "."
            if column > 0 and row > 0:
                self.matrix[row - 1, column - 1] = "."
            if column < 9 and row > 0:
                self.matrix[row - 1, column + 1] = "."
            if column > 0 and row < 9:
                self.matrix[row + 1, column - 1] = "."
            if column < 9 and row < 9:
                self.matrix[row + 1, column + 1] = "."
        # l - its similiar to top, only top, left and bottom are filled with water
        elif letter.capitalize() == "L":
            if row > 0:
                self.matrix[row - 1, column] = "."
            if column > 0:
                self.matrix[row, column - 1] = "."
            if row < 9:
                self.matrix[row + 1, column] = "."
            if column > 0 and row > 0:
                self.matrix[row - 1, column - 1] = "."
            if column > 0 and row < 9:
                self.matrix[row + 1, column - 1] = "."
        # R - top, right and bottom are filled with water
        elif letter.capitalize() == "R":
            if row > 0:
                self.matrix[row - 1, column] = "."
            if column < 9:
                self.matrix[row, column + 1] = "."
            if row < 9:
                self.matrix[row + 1, column] = "."
            if column < 9 and row > 0:
                self.matrix[row - 1, column + 1] = "."
            if column < 9 and row < 9:
                self.matrix[row + 1, column + 1] = "."

    """Insere uma parte de um navio"""
    def insert_ship_part(self, row, collum, letter):
        if self.matrix[row, collum] != "0":
            raise ValueError
        if letter == "C":
            self.BoatSizes[1] -= 1
        elif letter == "M":
            self.boatParts["M"] -=1
        else:
            self.boatParts["O"] -= 1
        self.matrix[row, collum] = letter.lower()

    def insert_ship(self, action):
        print(action)
        counter = action[0]
        print(counter, self.BoatSizes[counter])
        if self.BoatSizes[counter] <= 0:
            raise ValueError
        
        for i in range(1, counter + 1):     #seleciona cada tuplo com (x, (row, col, letter) * xVezes, ...)
            self.insert_ship_part(action[i][0], action[i][1], action[i][2])
        
        self.BoatSizes[counter] -= 1

    def print_Board(self):
        """Imprime o tabuleiro no standard output."""
        board_string = ""
        for i in range(len(self.row)):
            for j in range(len(self.row)):
                board_string += self.matrix[i, j]
                board_string += "\t"
            board_string += "\n"
        print(board_string)

    def fit_boat_of_two(self, row: int, column: int, letter: str):
        """Verifica se é possivel meter uma peça de tamanho dois numa linha com tamanho dois"""
        if letter == "T" and self.collum[column] == 2:
            self.matrix[row, column] = "T"
            self.matrix[row + 1, column] = "b"
            self.piece_water_spaces(row + 1, column, "B")
            self.collum[column] = 0
            self.row[row] -= 1
            self.row[row + 1] -= 1
        if letter == "B" and self.collum[column] == 2:
            self.matrix[row, column] = "B"
            self.matrix[row - 1, column] =  "t"
            self.piece_water_spaces(row - 1, column, "T")
            self.collum[column] = 0
            self.row[row] -= 1
            self.row[row - 1] -= 1
        if letter == "L" and self.row[row] == 2:
            self.matrix[row, column] = "L"
            self.matrix[row, column + 1] = "r"
            self.piece_water_spaces(row, column + 1, "R")
            self.row[row] = 0
            self.collum[column] -= 1
            self.row[column + 1] -= 1
        if letter == "R" and self.row[row] == 2:
            self.matrix[row, column] = "R"
            self.matrix[row, column - 1] = "l"
            self.piece_water_spaces(row, column - 1, "L")
            self.row[row] = 0
            self.collum[column] -= 1
            self.row[column - 1] -= 1

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
        collumn = [int(x.strip()) for x in stdin.readline().split("\t")[1:]]
        hints = int(stdin.readline())
        hints_list = []
        for i in range(hints):
            temp = stdin.readline().split("\t")
            hints_list.append(
                [int(x.strip()) if x.strip().isdigit() else x.strip() for x in temp[1:]]
            )
        new_board = Board(row, collumn)
        for i in range(hints):
            if hints_list[i][2] == "C":
                new_board.matrix[hints_list[i][0], hints_list[i][1]] = "C"
                new_board.piece_water_spaces(
                hints_list[i][0], hints_list[i][1], hints_list[i][2]
            )
                pass
            new_board.piece_water_spaces(
                hints_list[i][0], hints_list[i][1], hints_list[i][2]
            )
            new_board.fit_boat_of_two(hints_list[i][0], hints_list[i][1], hints_list[i][2])

        return new_board


class Bimaru(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        board.fill_water()
        #board.print_Board()
        self.inicial_state = BimaruState(board)

    def actions_4_boat(self, state: BimaruState, actions_list):
        if state.board.BoatSizes[4] != 0 and state.board.boatParts["M"] >= 2 and state.board.boatParts["O"] >= 2:    #faltam barcos de 4?                                      

            for row_i in range(len(state.board.row)):                   #percorrer as rows com os numeros de barcos
                if state.board.row[row_i] >= 4:                         #existe alguma row para por um barco de 4 peças?
                    for col_j in range(10):
                        # vamos verificar se:
                        #       se a coluna é menor que 6 (para não exceder o board)
                        #       podemos colocar uma peça nessa coluna
                        #       essa posição está vazia
                        if col_j <= 6\
                            and state.board.collum[col_j] >= 1 and state.board.matrix[row_i, col_j] == "0"\
                            and state.board.collum[col_j + 1] >= 1 and state.board.matrix[row_i, col_j + 1] == "0"\
                            and state.board.collum[col_j + 2] >= 1 and state.board.matrix[row_i, col_j+ 2] == "0"\
                            and state.board.collum[col_j + 3] >= 1 and state.board.matrix[row_i, col_j + 3] == "0":

                            actions_list.append((4,(row_i, col_j,"L"),(row_i, col_j + 1,"M"),(row_i, col_j + 2,"M"),(row_i, col_j + 3,"R")))

            for col_k in range(len(state.board.collum)):            #percorrer as collums com os numeros de barcos
                if state.board.collum[col_k] >= 4:                  #existe alguma collum para por um barco de 4?
                    for row_l in range(10):
                        # vamos verificar se:
                        #       se a linha é menor que 6 (para não exceder o board)
                        #       podemos colocar uma peça em cada linha
                        #       cada posição está vazia
                        if row_l <= 6 and state.board.row[row_l] >= 1 and state.board.matrix[row_l, col_k] == "0"\
                            and state.board.row[row_l + 1] >= 1 and state.board.matrix[row_l + 1, col_k] == "0"\
                            and state.board.row[row_l + 2] >= 1 and state.board.matrix[row_l + 2, col_k] == "0"\
                            and state.board.row[row_l + 3] >= 1 and state.board.matrix[row_l + 3, col_k] == "0":

                            actions_list.append((4,(row_l, col_k,"T"),(row_l + 1, col_k,"M"),(row_l + 2, col_k,"M"),(row_l + 3, col_k,"B")))

    def actions_3_boat(self, state: BimaruState, actions_list):
        if state.board.BoatSizes[3] != 0 and state.board.boatParts["M"] >= 1 and state.board.boatParts["O"] >= 2:           #faltam barcos de 3?
            for row_i in range(len(state.board.row)):                                   #percorrer as rows com os numeros de barcos
                 if state.board.row[row_i] >= 3:                                        #posso colocar um barco de 3?
                     for col_j in range(10):
                        #verificar se:
                        #       não estou no fim do board
                        #       posso colocar uma peça em cada coluna
                        #       cada posição está vazia
                        if col_j <= 7\
                        and state.board.collum[col_j] >= 1 and state.board.matrix[row_i, col_j] == "0"\
                        and state.board.collum[col_j + 1] >= 1 and state.board.matrix[row_i, col_j + 1] == "0"\
                        and state.board.collum[col_j + 2] >= 1 and state.board.matrix[row_i, col_j + 2] == "0":
                            
                            actions_list.append((3, (row_i, col_j, "L"), (row_i, col_j + 1, "M"), (row_i, col_j + 2, "R")))

            for col_k in range(len(state.board.collum)):                            #percorrer as colunas com os numeros dos barcos
                 if state.board.collum[col_k] >= 3:                                 #posso colocar um barco de 3?
                     for row_l in range(10):
                        #verificar se:
                        #       não estou no fim do board
                        #       posso colocar uma peça em cada linha
                        #       cada posição está vazia
                        if row_l <= 7\
                        and state.board.row[row_l] >= 1 and state.board.matrix[row_l, col_k] == "0"\
                        and state.board.row[row_l + 1] >= 1 and state.board.matrix[row_l + 1, col_k] == "0"\
                        and state.board.row[row_l + 2] >= 1 and state.board.matrix[row_l + 2, col_k] == "0":
                            
                            actions_list.append((3, (row_l, col_k, "T"), (row_l + 1, col_k, "M"), (row_l + 2, col_k, "B")))
            
    def actions_2_boat(self, state: BimaruState, actions_list):
        if state.board.BoatSizes[2] != 0 and state.board.boatParts["O"] >= 2:                   #faltam barcos de 2?
            for row_i in range(len(state.board.row)):           #percorrer as linhas com o numero de barcos
                if state.board.row[row_i] >= 2:                 #posso colocar um barco de 2?
                    for col_j in range(10):   
                        #verificar se:
                        #       não estou no fim do board
                        #       posso colocar uma peça em cada coluna
                        #       se a posição está vazia                  
                        if col_j <= 8\
                            and state.board.collum[col_j] >= 1 and state.board.matrix[row_i, col_j] == "0"\
                            and state.board.collum[col_j + 1] >= 1 and state.board.matrix[row_i, col_j + 1] == "0":
                            
                            actions_list.append((2,(row_i, col_j,"L"),(row_i, col_j + 1,"R")))
            
            for col_k in range(len(state.board.collum)):        #percorrer as colunas com o numero de barcos
                if state.board.row[col_k] >= 2:                 #posso colocar um barco de 2?
                    for row_l in range(10):
                        #verificar se:
                        #       não estou no fim do board
                        #       posso colocar uma peça em cada linha
                        #       se a posição está vazia      
                        if row_l <= 8\
                        and state.board.row[row_l] >= 1 and state.board.matrix[row_l, col_k] == "0"\
                        and state.board.row[row_l + 1] >= 1 and state.board.matrix[row_l + 1, col_k] == "0":
                            actions_list.append((2,(row_l, col_k,"T"),(row_l + 1, col_k,"B")))
    
    def actions_1_boat(self, state: BimaruState, actions_list):
        if state.board.BoatSizes[1] != 0:
            for row_i in range(len(state.board.row)):           #percorrer as linhas com o numero de barcos
                 if state.board.row[row_i] >= 1:                #posso colocar um barco de 1?
                     for col_j in range(10):
                        #verificar se:
                        #       posso colocar uma peça na coluna
                        #       se a posição está vazia      
                        if state.board.collum[col_j] >= 1 and state.board.matrix[row_i, col_j] == "0":
                            actions_list.append((1, (row_i, col_j, "C")))

    def actions(self, state: BimaruState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions_list = []

        self.actions_4_boat(state, actions_list)
        self.actions_3_boat(state, actions_list)
        self.actions_2_boat(state, actions_list)
        self.actions_1_boat(state, actions_list)

        return actions_list
            

    def result(self, state: BimaruState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        state.board.insert_ship(action)
        state.board.print_Board()
        return BimaruState(state.board)

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
    novo_ola = ola
    new_problem = Bimaru(ola)
    counter = 4
    for i in new_problem.actions(new_problem.inicial_state):
        even_newer_problem = Bimaru(ola)
        if int(i[0]) == counter:
            print("\nBARCOS DE " + str(counter) + "-----------")
            counter -= 1
        print(i)
        even_newer_problem.result(even_newer_problem.inicial_state, i)
        
    # TODO:
    # Ler o ficheiro do standard input,
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
