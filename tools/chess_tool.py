# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from chessimg2pos import predict_fen
from stockfish import Stockfish
import chess

# Defining the ChessTool class which extends BaseTool
class ChessTool(BaseTool):
    name : str = "chess_tool"
    description : str = "Given the path of an image, this tool returns the best next move that can be done on the chessboard. You must give ONLY the PATH of the image here! Pass in input b or w as color_turn based on whose turn is it. Use w if unspecified."

    def _run(self, img_path: str, color_turn: str) -> str:
        # Method to analyze the chessboard image and return the best move
        # Get the FEN string
        fen = predict_fen("./downloaded_files/image.png")  # Predicting the FEN string from the chessboard image

        # The fen predicted is always with a1 at the bottom left.
        # If it's black turn than the bottom left is h8, you need to reverse the positions retrieved.
        if color_turn == "b":
            ranks = fen.split('/')
            rotated_matrix = []
            for old_row in reversed(ranks):
                rotated_matrix.append(list(reversed(old_row)))
            final_fen = "/".join(["".join(row) for row in rotated_matrix])
            for length in reversed(range(2, 9)):
                final_fen = final_fen.replace(length * "1", str(length))
        else:
            final_fen = fen

        fen = f"{final_fen} {color_turn} - - 0 1"

        try:
            # Initializing Stockfish chess engine
            stockfish = Stockfish(path="C:/Users/FORMAGGA/Documents/personal/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")

            stockfish.set_fen_position(fen)

            next_move = str(stockfish.get_best_move())  # Getting the best move from Stockfish
        except Exception as e:
            print("Exception", e)
            raise e
        
        piece = stockfish.get_what_is_on_square(next_move[:2])  # Getting the piece on the starting square of the move

        next_move_fen = piece.name + next_move[2:]  # Constructing the FEN representation of the move

        return next_move_fen  # Returning the best move in FEN format