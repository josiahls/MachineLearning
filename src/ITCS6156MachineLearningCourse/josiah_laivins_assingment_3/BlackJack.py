import os

import numpy as np

from Logging import Logging

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def draw_card():
    return deck[np.random.randint(1, 13)]


class Player():
    """

        Player class to create a player object.
        eg: player = Player("player1", start_balance = 10, is_computer = 0)
        Above declaration will be for your agent. Only one non-computer
        player allowed in the local poker instance.
        However, you can create as many as you want computer_players
        (test for 3 players but you can try more) i.e
        computer = Player("computer", start_balance = 10, is_computer = 1).
        All the player names should be unique or else you will get error.

    """

    def __init__(self, player_name, start_balance=10, is_computer=0):
        self.player_name = player_name
        self.card = None
        self.total_balance = start_balance
        self.current_bet = 0
        self.is_computer = is_computer
        self.is_active = 0
        self.number_of_turn = 0
        self.match_bet = False

    def get_info(self):
        print("Player name: {}".format(self.player_name))
        print("Player card: {}".format(self.card))
        print("Player total_balance: {}".format(self.total_balance))
        print("Player is_active: {}".format(self.is_active))


class Poker:
    """
        Poker class will be the environment.
        Important functions for users:
        1. Init Poker class i.e poker = Poker(number_of_player = 2)
           <Please dont add more than the number you declare here)
        2. add_player(player1) - adds player1 object to the game.
        3. get_current_state - gets the current state of the env
           {'other_stats': [[9, 1] ,[9, 1]], 'total_pot_balance': 2, 'player_stats': [6, 9, 1]}
           other_stats will contain the opponent information's balance and their current bet
           total_pot_balance will have total bet played by all players
           player_stats will contain player_card, player total balance, player current bet
        4. get_valid_actions -  returns the valid action available for the player
           based on balance and which round the player is in. You should take actions
           based on the return value.


    """

    def __init__(self, number_of_player=2):
        self.all_players = []
        self.total_players = number_of_player
        self.total_pot_balance = 0
        self.player = None
        self.actions = [0, 1, 2, 3]
        self.number_of_turn = 0
        self.max_bet = 0

    def add_player(self, player_class):
        if player_class not in self.all_players and self.total_players > 0:
            self.all_players.append(player_class)
            self.total_players -= 1
            if player_class.is_computer == 0:
                if self.player == None:
                    self.player = player_class
                else:
                    raise Exception("Will override current player ! Only one non-computer player allowed !")
        else:
            raise Exception(
                "Maximum number of player allowed: {}. You can increase the player count while initializing the environment".format(
                    len(self.all_players)))

    def check_and_remove_players(self):
        player_remove, game_end = [], False
        for _player in self.all_players:
            if _player.total_balance <= 1:
                player_remove.append(_player)
        if len(player_remove) > 0:
            if self.player not in player_remove:
                for player in player_remove:
                    print("Removing player: {} due to insufficient funds !!".format(player.player_name))
                    self.all_players.remove(player)
            else:
                print("Player doesnt have balance !")
                game_end = True

    def deal(self):
        reset_game = False
        self.total_pot_balance = 0
        self.number_of_turn = 0

        if self.check_and_remove_players():
            reset_game = True
        else:
            if len(self.all_players) == 1:
                reset_game = True
            else:
                for _player in self.all_players:
                    if _player.total_balance > 1:
                        _player.card = draw_card()
                        _player.total_balance -= 1
                        _player.current_bet = 1
                        _player.is_active = 1
                        _player.number_of_turn = 0
                        _player.match_bet = False
                        self.total_pot_balance += 1
                        self.max_bet = 0
                    else:
                        print("{} can no longer play the game !".format(_player.player_name))
                        reset_game = True
        return reset_game

    def get_current_state(self):
        current_state_dict = {}
        current_state_dict['other_stats'] = [[_player.total_balance, _player.current_bet] for _player in
                                             self.all_players if _player != self.player]
        current_state_dict['total_pot_balance'] = self.total_pot_balance
        current_state_dict['player_stats'] = [self.player.card, self.player.total_balance, self.player.current_bet]
        return current_state_dict

    def get_valid_actions(self, player_name):
        list_player = [_player for _player in self.all_players if _player.player_name == player_name]
        if len(list_player) == 1:
            player = list_player[0]
            if player.number_of_turn == 0:
                if player.total_balance > 3:
                    actions = [0, 1, 2, 3]
                elif player.total_balance <= 3:
                    if player.total_balance <= 1:
                        actions = [0]
                    elif player.total_balance > 1 and player.total_balance <= 2:
                        actions = [0, 1]
                    else:
                        actions = [0, 1, 2]
            else:
                possible_max_bet = player.total_balance - (self.max_bet - (player.current_bet - 1))
                if possible_max_bet >= 2:
                    actions = [0, self.max_bet - (player.current_bet - 1)]
                else:
                    actions = [0]
            return actions
        else:
            raise Exception("Invalid player name! Use the player name defined while initialzing the environment")

    def print_actions(self, player, action):
        if action == 0:
            print("{} folds".format(player))
        else:
            print("{} plays {}.".format(player, action))

    def computer_play(self):
        list_player = [_player for _player in self.all_players if
                       _player.player_name != self.player.player_name and _player.is_active == 1 and _player.match_bet == False]
        for player in list_player:
            actions_available = self.get_valid_actions(player.player_name)
            print("Available actions for: {} are {}".format(player.player_name, actions_available))
            action_taken = actions_available[np.random.randint(len(actions_available))]
            self.print_actions(player.player_name, action_taken)
            if self.max_bet == 0 and action_taken != 0 or self.max_bet < action_taken:
                self.max_bet = action_taken
            self.take_action(player.player_name, action_taken)

    def get_player(self, player_name):
        return_player = [player for player in self.all_players if player.player_name == player_name]
        if len(return_player) != 1:
            print("Invalid Player")
            return None
        else:
            return return_player[0]

    def take_action(self, player_name, action_taken):
        player = self.get_player(player_name)
        player.number_of_turn += 1
        if action_taken != 0:
            player.total_balance -= action_taken
            player.current_bet += action_taken
            self.total_pot_balance += action_taken
            player.is_active = 1
        else:
            player.is_active = 0

    def check_game(self):
        game_over = True
        if np.sum([_player.is_active for _player in self.all_players if _player != self.player]) > 0:
            game_over = False
        return game_over

    def return_winner(self, players):
        max_card = np.max([_player.card for _player in players])
        return [_player for _player in players if _player.card == max_card]

    def settle_balance(self, winner):
        reward = 0
        if winner == "draw":
            active_player = [_player for _player in self.all_players if _player.is_active]
            if len(active_player) == 0:
                print("Active player cards: {}".format(
                    [(_player.player_name, _player.card) for _player in self.all_players]))
                winning_players = self.all_players
            else:
                print("Active player cards: {}".format(
                    [(_player.player_name, _player.card) for _player in self.all_players if _player.is_active]))
                winning_players = self.return_winner(active_player)

            per_player_share = self.total_pot_balance / len(winning_players)
            for _player in winning_players:
                _player.total_balance += per_player_share
            if self.player in winning_players:
                reward = per_player_share
            else:
                reward = -self.player.current_bet

        if winner == "player":
            self.player.total_balance += self.total_pot_balance
            reward = self.total_pot_balance
            winning_players = [self.player]

        if winner == "computer":
            active_player = [_player for _player in self.all_players if _player != self.player and _player.is_active]
            winning_players = self.return_winner(active_player)
            per_player_share = self.total_pot_balance / len(winning_players)
            for _player in winning_players:
                _player.total_balance += per_player_share
            reward = -self.player.current_bet

        print("Printing winning players: {}".format(' '.join([player.player_name for player in winning_players])))
        print("Reward : {}".format(reward))
        return reward

    def check_game_return_reward(self, round_num):
        print("Check results for Round Number: {}".format(round_num))
        if self.check_game() and self.player.is_active == 0:
            print("Computer folds, Player folds !")
            game_over = True
            winner = "draw"
            reward = 0
            return game_over, reward, winner

        elif self.check_game() and self.player.is_active:
            print("Computer folds, Player active !")
            game_over = True
            reward = self.total_pot_balance
            winner = "player"
            return game_over, reward, winner

        elif not self.check_game() and self.player.is_active == 0:
            print("Computer active, Player folds !")
            game_over = True
            reward = -self.player.current_bet
            winner = "computer"
            return game_over, reward, winner

        if round_num == 1:
            print("Computer active, Player active ! ")
            game_over = False
            reward = 0
            winner = "draw"
            return game_over, reward, winner

        else:
            print("Computer active, Player active in Round 2 ! End game now !")
            game_over = True
            active_players = [player for player in self.all_players if player.is_active]
            print("Active player cards: {}".format([(_player.player_name, _player.card) for _player in active_players]))
            winning_players = self.return_winner(active_players)
            if self.player in winning_players:
                if len(winning_players) > 1:
                    reward = self.total_pot_balance / len(winning_players)
                    winner = "draw"
                else:
                    reward = self.total_pot_balance
                    winner = "player"
            else:
                reward = -self.player.current_bet
                winner = "computer"

            return game_over, reward, winner

    def update_match(self):
        for player in [player for player in self.all_players if player.current_bet - 1 == self.max_bet]:
            player.match_bet = True

    def check_computer_status(self):
        computer_play = False
        if len([player for player in self.all_players if
                player.is_computer == 1 and player.match_bet == False and player.is_active == 1]) > 0:
            computer_play = True
        return computer_play

    def player_play(self, player_name, action_taken):
        game_over = False
        reward = 0
        winner = None
        self.number_of_turn += 1

        if self.number_of_turn == 1:
            if self.max_bet == 0 and action_taken != 0 or self.max_bet < action_taken:
                self.max_bet = action_taken
            self.print_actions(player_name, action_taken)
            self.computer_play()
            self.take_action(player_name, action_taken)
            self.update_match()
            if self.player.is_active:
                if self.player.match_bet and not self.check_computer_status():
                    print("Player and Computer both Bet Max and match !")
                    _, reward, winner = self.check_game_return_reward(round_num=1)
                    game_over = True
            else:
                if self.check_computer_status():
                    print("Player folds, computer players are active, they will play among themselves !")
                    self.computer_play()
                game_over, reward, winner = self.check_game_return_reward(round_num=1)

            if game_over:
                print("Winner: {}".format(winner))
                new_reward = self.settle_balance(winner)
                if winner == "draw":
                    reward = new_reward
            else:
                if self.player.match_bet and self.check_computer_status():
                    print("Player Bet Max, computer will match now !")
                    self.computer_play()
                    game_over, reward, winner = self.check_game_return_reward(round_num=2)
                    print("Winner: {}".format(winner))
                    new_reward = self.settle_balance(winner)
                    if winner == "draw":
                        reward = new_reward

            return self.get_current_state(), reward, game_over

        if self.number_of_turn == 2:
            self.print_actions(player_name, action_taken)
            self.take_action(player_name, action_taken)
            if self.check_computer_status():
                self.computer_play()
            game_over = True
            _, reward, winner = self.check_game_return_reward(round_num=2)
            print("Winner: {}".format(winner))
            new_reward = self.settle_balance(winner)
            if winner == "draw":
                reward = new_reward

        return self.get_current_state(), reward, game_over


class PokerEnvWrapper(object):
    def __init__(self, poker_env: Poker, player_name):
        # Init the poker envionment
        self.poker_env = poker_env
        # Init the states
        self.player_name = player_name
        self.state = self.get_cur_state()

        # Init the actions
        self.n_a = len(self.poker_env.actions)
        self.actions = self.poker_env.actions

        # Init the size. This needs to be more complex
        self.size = self.state
        # The total pot balance could be any of these except...
        self.size[:] = max(self.state) * len(self.poker_env.all_players)
        self.size[-3] = 12  # This is the slot for holding the player's current card
        self.size[-4] = max(self.state)
        self.size += 2
        self.size = tuple(self.size)

    def convert_poker_size(self):
        pass

    def get_size(self):
        return self.size

    def get_actions(self):
        return self.poker_env.get_valid_actions(self.player_name)

    def exclude_invalid_regions(self, Q):
        return Q

    def get_cur_state(self):
        poker_state = self.poker_env.get_current_state()
        state = np.array(poker_state['other_stats']).flatten()
        state = np.hstack((state, poker_state['total_pot_balance']))
        # noinspection PyComparisonWithNone,PyComparisonWithNone
        poker_state['player_stats'][poker_state['player_stats'] == None] = 0
        state = np.hstack((state, poker_state['player_stats']))
        return state.astype(int)

    def init(self, start):
        poker = Poker(number_of_player=2)
        player1 = Player("prajval", start_balance=10, is_computer=0)
        computer1 = Player("computer1", start_balance=10, is_computer=1)
        # computer2 = Player("computer2", start_balance=10, is_computer=1)
        poker.add_player(player1)
        poker.add_player(computer1)
        # poker.add_player(computer2)
        self.poker_env = poker

    def next(self, action):
        self.poker_env.take_action(self.player_name, action)
        return self.poker_env.player_play(self.player_name, action)[1]

    def is_goal(self):
        return False

    def deal(self):
        return self.poker_env.deal()


if __name__ == '__main__':
    from PokerQLearningModel import RLAgent
    poker = Poker(number_of_player=2)
    player1 = Player("prajval", start_balance=10, is_computer=0)
    computer1 = Player("computer1", start_balance=10, is_computer=1)
    # computer2 = Player("computer2", start_balance=10, is_computer=1)
    poker.add_player(player1)
    poker.add_player(computer1)
    # poker.add_player(computer2)

    rl = RLAgent(PokerEnvWrapper(poker, "prajval"))
    iterations = 100000

    # Try to load a previous Q
    try:
        rl.Q = np.load(f'poker_q_{iterations}.npy')
        rtrace = np.load(f'rtrace_{iterations}.npy')
        steps = np.load(f'steps_{iterations}.npy')
    except IOError:
        rtrace, steps = rl.train_sarsa(start=None, poker=poker, gamma=.99, alpha=.01,
                                   epsilon=0.1, maxiter=iterations)
        np.save(f'poker_q_{iterations}', rl.Q)
        np.save(f'rtrace_{iterations}', rtrace)
        np.save(f'steps_{iterations}', steps)

    # Logging.plot_train(rl, rtrace, steps)
    while not poker.deal():
        print("-" * 50)
        print("Deal Start!")
        print("Start State: {}".format(poker.get_current_state()))
        actions = poker.get_valid_actions("prajval")
        print("Available actions for: prajval are {}".format(actions))
        action_taken = rl.greedy(rl.env.get_cur_state())
        result = poker.player_play("prajval", action_taken)
        while not result[-1]:
            actions = poker.get_valid_actions("prajval")
            print("Available actions for: prajval are {}".format(actions))
            action_taken = rl.greedy(rl.env.get_cur_state())
            result = poker.player_play("prajval", action_taken)
        print("Final Result: {}".format(result))
        print("*" * 50)

    wins, loses = rl.test(maxstep=iterations)
    print(f'Wins: {np.sum(wins)} Loses: {np.sum(loses)}')
    # Logging.plot_win_loss(rl, loses, wins, steps, rl.env)
