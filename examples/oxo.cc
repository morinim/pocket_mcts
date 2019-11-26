/**
 *  \file
 *  \remark This file is part of POCKET MCTS.
 *
 *  \copyright Copyright (C) 2019 Manlio Morini.
 *
 *  \license
 *  This Source Code Form is subject to the terms of the Mozilla Public
 *  License, v. 2.0. If a copy of the MPL was not distributed with this file,
 *  You can obtain one at http://mozilla.org/MPL/2.0/
 *
 *  Example game state class for Tic-Tac-Toe
 */

#include <array>
#include <fstream>

#include "pocket_mcts.h"

class action
{
public:
  constexpr explicit action(int v) noexcept : val_(v) {}

  operator unsigned() const noexcept { return val_; }

private:
  int val_;
};

constexpr unsigned PLAYER_X = 0;
constexpr unsigned PLAYER_O = 1;
constexpr unsigned EMPTY    = 2;

class oxo_state
{
public:
  using action = ::action;

  oxo_state() : player_(PLAYER_X), board_{EMPTY, EMPTY, EMPTY,
                                          EMPTY, EMPTY, EMPTY,
                                          EMPTY, EMPTY, EMPTY} {}

  void take_action(action a)
  {
    assert(board_[a] == EMPTY);

    board_[unsigned(a)] = player_;
    player_   = !player_;
  }

  std::vector<action> actions() const
  {
    if (is_final())
      return {};

    std::vector<action> ret;

    for (unsigned i(0); i < board_.size(); ++i)
      if (board_[i] == EMPTY)
        ret.push_back(action(i));

    return ret;
  }

  std::vector<double> eval() const
  {
    static const std::vector<std::vector<unsigned>> sequences =
    {
      {0,1,2}, {3,4,5}, {6,7,8}, {0,3,6}, {1,4,7}, {2,5,8}, {0,4,8}, {2,4,6}
    };

    for (const auto &s : sequences)
      if (board_[s[0]] == board_[s[1]] && board_[s[1]] == board_[s[2]]
          && board_[s[0]] != EMPTY)
      {
        if (board_[s[0]])
          return {0.0, 1.0};
        else
          return {1.0, 0.0};
      }

    if (std::find(board_.begin(), board_.end(), EMPTY) == board_.end())
      return {0.5, 0.5};

    return {-1.0, -1.0};
  }

  bool is_final() const
  {
    const auto score(eval());
    return score[0] >= 0.0;
  }

  unsigned agent_id() const { return player_; }

  friend std::ostream &operator<<(std::ostream &o, const oxo_state &s)
  {
    const auto symb([](unsigned v)
                    { return v == PLAYER_X ? 'X'
                           : v == PLAYER_O ? 'O'
                           : '.'; });
    return o << symb(s.board_[0]) << symb(s.board_[1]) << symb(s.board_[2])
             << '\n'
             << symb(s.board_[3]) << symb(s.board_[4]) << symb(s.board_[5])
             << '\n'
             << symb(s.board_[6]) << symb(s.board_[7]) << symb(s.board_[8])
             << '\n';
  }

private:
  unsigned player_;

  std::array<decltype(player_), 9> board_;  // Arrangement of the squares:
                                            //     012
                                            //     345
                                            //     678
};

int main()
{
  oxo_state s;

  for (;;)
  {
    std::cout << s;
    std::cout << "\nYour move: ";

    unsigned a;
    std::cin >> a;

    s.take_action(oxo_state::action(a));

    std::cout << s << std::endl;

    if (s.is_final())
      return 0;

    using mcts = pocket_mcts::uct<oxo_state>;

    std::ofstream log("log.txt");

    const auto & [move, scores] = mcts(s)
                                  .max_iterations(1000)
                                  .log(log).log_depth(2).verbose(true)
                                  .run();

    if (move)
    {
      s.take_action(*move);

      std::cout << "My move: " << *move << " (";
      for (auto v : scores)
        std::cout << ' ' << v;
      std::cout << ")\n";
    }

    if (s.is_final())
      return 0;
  }
}
