import unittest
from bimaru import Board

class TestBoard(unittest.TestCase):
    def test_get_value(self):
        board = Board([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        board.matrix[0][0] = 'C'
        board.matrix[1][1] = 'T'
        board.matrix[2][2] = 'M'
        board.matrix[3][3] = 'B'
        self.assertEqual(board.get_value(0, 0), "C")
        self.assertEqual(board.get_value(1, 1), "T")
        self.assertEqual(board.get_value(2, 2), "M")
        self.assertEqual(board.get_value(3, 3), "B")

    def test_adjacent_vertical_values(self):
        board = Board([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        board.matrix[0][0] = 'C'
        board.matrix[1][0] = 'T'
        board.matrix[2][0] = 'M'
        board.matrix[3][0] = 'B'
        self.assertEqual(board.adjacent_vertical_values(1, 0), ("C", "M"))
        self.assertEqual(board.adjacent_vertical_values(2, 0), ("T", "B"))
        self.assertEqual(board.adjacent_vertical_values(0, 0), (None, "T"))
        self.assertEqual(board.adjacent_vertical_values(3, 0), ("M", None))

    def test_adjacent_horizontal_values(self):
        board = Board([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        board.matrix[0][0] = 'C'
        board.matrix[0][1] = 'T'
        board.matrix[0][2] = 'M'
        board.matrix[0][3] = 'B'
        self.assertEqual(board.adjacent_horizontal_values(0, 1), ("C", "M"))
        self.assertEqual(board.adjacent_horizontal_values(0, 2), ("T", "B"))
        self.assertEqual(board.adjacent_horizontal_values(0, 0), (None, "T"))
        self.assertEqual(board.adjacent_horizontal_values(0, 3), ("M", None))

    def test_fill_water(self):
        board = Board([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        board.matrix[0][0] = "C"
        board.matrix[1][0] = "T"
        board.matrix[2][0] = "M"
        board.matrix[3][0] = "B"
        board.row[0] = 0
        board.collum[0] = 0
        board.fill_water()
        self.assertEqual(board.matrix[0][1], ".")
        self.assertEqual(board.matrix[0][2], ".")
        self.assertEqual(board.matrix[0][3], ".")
        self.assertEqual(board.matrix[1][0], "T")
        self.assertEqual(board.matrix[2][0], "M")
        self.assertEqual(board.matrix[3][0], "B")

if __name__ == '__main__':
    unittest.main()