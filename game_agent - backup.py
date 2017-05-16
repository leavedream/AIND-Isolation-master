"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    
    #return heuristic_space(game, player)
    
    # Aggresive: apply factor of 2 to opponent moves
    #return heuristic_move_diff(game, player, 1, 2)

    # Aggresive: regular heuristic function (#my_moves - #oppt_move)
    return heuristic_move_divide(game, player)
    
    # Aggresive: regular heuristic function (#my_moves - #oppt_move)
    #return heuristic_move_diff(game, player, 1, 0.5)
    
    
    
def heuristic_move_diff(game, player, factor1 = 1, factor2 = 1):
    """Calculate the heuristic value of a game state from the point of view
    of the given player in an aggresive way.
    
    Apply factors to player and opponent's move 
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    
    factor1: integer, if not provided, default to 1
        The factor that will be applied to player's #_my_moves (factor1 * #my_moves)
    
    factor2: integer, if not provided, default to 1
        The factor that will be applied to player's #_oppo_moves (factor2 * #oppo_move)
        
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = float(len(game.get_legal_moves(player)))
    score_opp = float(len(game.get_legal_moves(game.get_opponent(player))))
    return score * factor1 - score_opp * factor2    

def heuristic_move_divide(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player in an aggresive way.
    
    heurisitic function: #my_moves / #opponent_move 
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    score = float(len(game.get_legal_moves(player)))
    score_opp = float(len(game.get_legal_moves(game.get_opponent(player))))
    if score_opp == 0:
        return float("inf")
        
    return score / score_opp        

def heuristic_space(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player in an aggresive way.
    
    heurisitic function: #blank_spaces 
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    my_move = len(game.get_legal_moves(player))
    oppo_move = len(game.get_legal_moves(game.get_opponent(player)))
    
    if oppo_move == 0:
        return 100
    if my_move == 0:
        return -100
    
    area = float(game.width * game.height)
    blanks = float(len(game.get_blank_spaces))
    if blanks > area // 2:
        return my_move - oppo_move
        
    return my_move - 2 * oppo_move
    #return my_move - oppo_move


    
class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        # At the beginning of the game, always put the first move to the center of the board
        if game.move_count == 0:
            move = (game.height // 2, game.width // 2)
            return move
            
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self == game.__player_1__:
                maximizing_player = True
            else:
                maximizing_player = False
                
            if self.method == 'minimax':
                if self.iterative:
                    depth = 0
                    while self.time_left() > self.TIMER_THRESHOLD:
                        v, move = self.minimax(game, depth, maximizing_player)
                        depth = depth + 1
                else:
                    v, move = self. minimax(game, self.search_depth, maximizing_player)
                    
            if self.method == 'alphabeta':
                if self.iterative:
                    depth = 0
                    while self.time_left() > self.TIMER_THRESHOLD:
                        v, move = self.alphabeta(game, depth, maximizing_player)
                        depth = depth + 1
                else:
                    v, move = self.alphabeta(game, self.search_depth, maximizing_player)
                    
            # Return the best move from the last completed search iteration    
            return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            # randomly pick one from legal moves...
            legal_moves = game.get_legal_moves()
            if (len(legal_moves)>0):
                return legal_moves[0]

        

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
            
        if game.is_winner(self):
            return self.score(game, self), game.get_player_location(self)
        if game.is_loser(self):
            return self.score(game, self), (-1, -1)    
        if depth==0:
            
            
        # TODO: finish this function!
        legal_moves = game.get_legal_moves()
        if len(legal_moves)==0:
            return self.score(game, self), (-1, -1)
        if maximizing_player:
            v = float("-inf")
        else:
            v = float("+inf")
        move = legal_moves[0]
        for m in legal_moves:
            newgame = game.forecast_move(m)
            newscore, _ = self.minimax(newgame, depth-1, not maximizing_player)
            if (newscore > v and maximizing_player) or (newscore < v and (not maximizing_player)):
                v = newscore
                move = m
                
        return v, move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # TODO: finish this function!
        if depth==0:
            if game.is_winner(self):
                return self.score(game, self), game.get_player_location(self)
            if game.is_loser(self):
                return self.score(game, self), (-1, -1)
            
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return self.score(game, self), (-1, -1)
            
        if maximizing_player:
            v = float("-inf")
        else:
            v = float("+inf")
        move = legal_moves[0]
        for m in legal_moves:
            newgame = game.forecast_move(m)
            newscore, _ = self.alphabeta(newgame, depth-1, alpha, beta, not maximizing_player)
            if maximizing_player:
                if v < newscore:
                    v = newscore
                    move = m
                if v >= beta:
                    return v, move
                alpha = max(alpha, v)
            if not maximizing_player:
                if v > newscore:
                    v = newscore
                    move = m
                if v <= alpha:
                    return v, move
                beta = min(beta, v)
                
        return v, move

