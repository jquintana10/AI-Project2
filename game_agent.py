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
    raise NotImplementedError


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
        self.game = game
        self.legal_moves = legal_moves


        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if not legal_moves:
            return (-1, -1)

        try:

            if self.method == 'minimax':
                _, best_move = self.minimax(self.game, 1, True)

            if self.method == 'alphabeta':
                _, best_move = self.alphabeta(self.game, 1, True)

            return best_move
            #return minimax()
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            #pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        #raise NotImplementedError

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



        self.game = game
        self.depth = depth
        self.maximizing_player = maximizing_player
        legal_moves = game.get_legal_moves(self)


        def max_value(self, maxgame, maxdepth):

            maxlegal_moves = maxgame.get_legal_moves(self)

            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            print("depth:", depth)
            print("depth max:", depth)
            if maxdepth == depth:
                best_score = self.score(maxgame, self)
                print("sali")
                return best_score

            if not maxlegal_moves:
                best_score = self.score(maxgame, self)
                return best_score

            v = float("-inf")
            for move in maxlegal_moves:
                v = max(v, min_value(self, maxgame.forecast_move(move), maxdepth+1))

            #print("max v", v)

            return v

        def min_value(self, mingame, mindepth):

            minlegal_moves = mingame.get_legal_moves(self)

            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            print("depth:", depth)
            print("min depth:", mindepth)
            if mindepth == depth:
                best_score = self.score(mingame, self)
                print("sali:", best_score)
                return best_score

            print("Never")
            if not minlegal_moves:
                best_score = self.score(mingame, self)
                return best_score

            best_score = float("inf")
            print("MinLegalmove:", minlegal_moves)
            for move in minlegal_moves:
                print("MinMove:", move)
                best_score = min(best_score, max_value(self, mingame.forecast_move(move), mindepth+1))
            print("regrese con v:", best_score)
            return best_score


        # TODO: finish this function!
        if not legal_moves:
            return (-1, -1)

        score = float("-inf")
        print("Legal Move Principal", legal_moves)
        for m in legal_moves:
            print("move in principal:", m)
            v = min_value(self, game.forecast_move(m), 1)

            if (v > score):
                score = v
                move = m
            print("Score", score)
        return score, move """

        self.game = game
        self.depth = depth
        self.maximizing_player = maximizing_player
        legal_moves = game.get_legal_moves(self)

        # TODO: finish this function!
        if not legal_moves:
            return (-1, -1)

        score = float("-inf")
        for m in legal_moves:
            v = self.score(self.game.forecast_move(m),self)

            if (v > score):
                score = v
                move = m
            print("Score:", score)
            print("Move:", move)
        return score, move


        #raise NotImplementedError

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

         self.game = game
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.maximizing_player = maximizing_player
        legal_moves = game.get_legal_moves(self)

        def max_value(self, game, alpha, beta):

            maxlegal_moves = self.game.get_legal_moves(self)

            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            if not maxlegal_moves:
                best_score = self.score(self.game, self)
                return best_score

            v = float("-inf")
            for move in maxlegal_moves:
                v = max(v, min_value(self, game.forecast_move(move), alpha, beta))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(self, game, alpha, beta):

            minlegal_moves = self.game.get_legal_moves(self)

            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()

            if not minlegal_moves:
                best_score = self.score(self.game, self)
                return best_score

            v = float("inf")
            for move in minlegal_moves:
                v = min(v, max_value(self, game.forecast_move(move), alpha, beta))
                if v<= alpha:
                    return v
                beta = min(beta, v)
            return v


        # TODO: finish this function!
        if not legal_moves:
            return (-1, -1)

        if maximizing_player:
            score = float("-inf")
            for m in legal_moves:
                v = min_value(self, game.forecast_move(m), alpha, beta)
                if (v > score):
                    score = v
                    move = m

                    # score, move = max([(min_value(game.forecast_move(m),alpha, beta),m) for m in legal_moves])
        else:
            score = float("inf")
            for m in legal_moves:
                v = max_value(self, game.forecast_move(m), alpha, beta)
                if (v < score):
                    score = v
                    move = m
                    # score, move = min([(max_value(game.forecast_move(m),alpha, beta), m) for m in legal_moves])

        return score, move"""
