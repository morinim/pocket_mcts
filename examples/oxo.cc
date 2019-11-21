#include <fstream>

#include "pocket_mcts.h"

class action
{
public:
  constexpr explicit action(int v) noexcept : val_(v) {}

  static constexpr action sentry() noexcept { return action(9); }

  operator unsigned() const noexcept { return val_; }

private:
  int val_;
};

class oxo_state
{
public:
  using action = ::action;

  oxo_state() : player_(0), board_{2, 2, 2, 2, 2, 2, 2, 2, 2} {}

  void take_action(action a)
  {
    assert(board_[a] == 2);

    board_[a] = player_;
    player_   = !player_;
  }

  std::vector<action> actions() const
  {
    std::vector<action> ret;

    for (unsigned i(0); i < 9; ++i)
      if (board_[i] == 2)
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
          && board_[s[0]] != 2)
      {
        if (board_[s[0]])
          return {0.0, 1.0};
        else
          return {1.0, 0.0};
      }

    if (actions().empty())
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
                    { return v == 0 ? 'X' : v == 1 ? 'O' : '.'; });
    return o << symb(s.board_[0]) << symb(s.board_[1]) << symb(s.board_[2])
             << '\n'
             << symb(s.board_[3]) << symb(s.board_[4]) << symb(s.board_[5])
             << '\n'
             << symb(s.board_[6]) << symb(s.board_[7]) << symb(s.board_[8])
             << '\n';
  }

private:
  unsigned player_;    // `0` is 'X' player, `1` is 'O' player, `2` is a
                       // sentinel value
  unsigned board_[9];  // Squares have this arrangement:
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
    std::cout << "\nMove: ";

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

    s.take_action(move);

    for (auto v : scores)
      std::cout << ' ' << v;
    std::cout << std::endl;

    if (s.is_final())
      return 0;
  }
}
