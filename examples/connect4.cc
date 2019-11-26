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
 *  Example game state class for Connect 4
 */

#include <fstream>
#include <vector>

#include "pocket_mcts.h"

const std::string ansi_red_background = "\033[1;41m";
const std::string ansi_white_background = "\033[1;47m";
const std::string ansi_reset = "\033[0m";

const std::string p1 = ansi_red_background + "    " + ansi_reset;
const std::string p2 = ansi_white_background + "    " + ansi_reset;

constexpr std::size_t ROWS = 6;
constexpr std::size_t COLS = 7;

class action
{
public:
  constexpr explicit action(int v) noexcept : val_(v) {}

  operator unsigned() const noexcept { return val_; }

private:
  int val_;
};

class state
{
public:
  using action = ::action;

  state() : player_(0), bitboard_{0, 0}, height_{0, 0, 0, 0, 0, 0, 0} {}

  void take_action(action col)
  {
    assert(height_[col] < ROWS);

    const auto move(1ULL << (col*COLS + height_[col]));
    bitboard_[player_] |= move;

    ++height_[col];
    player_ = !player_;
  }

  std::vector<action> actions() const
  {
    if (is_final())
      return {};

    std::vector<action> ret;

    for (unsigned i(0); i < COLS; ++i)
      if (height_[i] < ROWS)
        ret.push_back(action(i));

    return ret;
  }

  std::vector<double> eval() const
  {
    const auto is_win([](std::uint64_t bb)
                      {
                        if (bb & (bb >> 6) & (bb >> 12) & (bb >> 18))
                          return true;  // diagonal tl-br
                        if (bb & (bb >> 8) & (bb >> 16) & (bb >> 24))
                          return true;  // diagonal bl-tr
                        if (bb & (bb >> 7) & (bb >> 14) & (bb >> 21))
                          return true;  // horizontal
                        if (bb & (bb >> 1) & (bb >>  2) & (bb >>  3))
                          return true;  // vertical
                        return false;
                      });

    if (is_win(bitboard_[0]))
      return {1.0, 0.0};

    if (is_win(bitboard_[1]))
      return {0.0, 1.0};

    for (unsigned i(0); i < COLS; ++i)
      if (height_[i] < ROWS)
        return {-1.0, -1.0};

    return {0.5, 0.5};
  }

  bool is_final() const
  {
    const auto score(eval());
    return score[0] >= 0.0;
  }

  unsigned agent_id() const { return player_; }

  friend std::ostream &operator<<(std::ostream &o, const state &s)
  {
    for (int r(ROWS); r--;)
    {
      std::string row_str("| "), sep_str("|");
      for (unsigned c(0); c < COLS; ++c)
      {
        const std::uint64_t index(1ULL << (c * COLS + r));

        if (s.bitboard_[0] & index)
          row_str += p1 + " | ";
        else if (s.bitboard_[1] & index)
          row_str += p2 + " | ";
        else
          row_str += "     | ";

        sep_str += "------|";
      }

      o << row_str << '\n' << row_str << '\n' << sep_str << '\n';
    }

    o << '|';
    for (unsigned c(0); c < COLS; ++c)
      o << "   " << c << "  |";

    return o << '\n';
  }

private:
  unsigned player_;     // `0` is 'X' player, `1` is 'O' player, `2` is a
                        // sentinel value

  std::uint64_t bitboard_[2];  // see https://github.com/denkspuren/BitboardC4/blob/master/BitboardDesign.md

  unsigned height_[COLS];
};

int read_move()
{
  std::cout << "\nYour move: ";

  int c;
  do
  {
    c = std::getchar();
  } while (c != '.' && !std::isdigit(c));

  if (c == '.')
    c = -1;
  else
    c = c - '0';

  return c;
}

bool legal(const state &s, state::action a)
{
  const auto actions(s.actions());
  return std::find(actions.begin(), actions.end(), a) != actions.end();
}

int main()
{
  state s;
  std::cout << s << std::endl;

  for (;;)
  {
    int a(-1);
    do
    {
      a = read_move();
      if (a < 0)
        return 0;
    } while (!legal(s, state::action(a)));

    s.take_action(state::action(a));
    std::cout << s << std::endl;

    if (s.is_final())
      return 0;

    using mcts = pocket_mcts::uct<state>;

    std::ofstream log("log.txt");

    const auto & [move, scores] = mcts(s)
                                  .max_iterations(10000)
                                  .log(log).log_depth(2).verbose(true)
                                  .run();

    if (move)
    {
      s.take_action(*move);

      std::cout << "My move: " << *move << " (";
      for (auto v : scores)
        std::cout << ' ' << v;
      std::cout << ")\n";

      std::cout << s << std::endl;
    }

    if (s.is_final())
      return 0;
  }
}
