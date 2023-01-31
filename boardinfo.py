from pprint import pprint

import brainflow
from brainflow.board_shim import BoardIds, BoardShim

board_id = BoardIds.CYTON_BOARD.value
pprint(BoardShim.get_board_descr(board_id))
