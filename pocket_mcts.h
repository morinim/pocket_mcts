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
 *  A minimal implementation of Monte Carlo Tree Search (MCTS) in C++17
 */

#if !defined(POCKET_MCTS_H)
#define      POCKET_MCTS_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <sstream>
#include <vector>

namespace pocket_mcts
{

using namespace std::chrono_literals;

namespace utility
{

///
/// \param[in] c a STL-like container
/// \return      a random element of `c`
///
template<class C>
const typename C::value_type &random_element(const C &c)
{
  assert(!c.empty());

  thread_local std::mt19937 random_engine;

  std::uniform_int_distribution<> d(0, c.size() - 1);

  return *std::next(c.begin(), d(random_engine));
}

///
/// Measures the time between two points.
///
/// The timer class cuts down the verbose syntax needed to measure the elapsed
/// time. It's similar to `boost::cpu_timer` but tailored to our needs (so less
/// general).
///
/// The simplest and most common use is:
///
///     int main()
///     {
///       timer t;
///
///       do_stuff_and_burn_some_time();
///
///       std::cout << "Elapsed (ms): " << t.elapsed().count() << '\n'
///     }
///
/// \warning
/// A useful recommendation is to never trust timings unless they are:
/// - at least 100 times longer than the CPU time resolution
/// - run multiple times
/// - run on release builds.
/// .. and results that are too good need to be investigated skeptically.
///
/// \remark
/// The original idea is of Kjellkod (http://kjellkod.wordpress.com).
///
class timer
{
public:
  timer() noexcept : start_(std::chrono::steady_clock::now()) {}

  void restart() noexcept { start_ = std::chrono::steady_clock::now(); }

  /// \return time elapsed in milliseconds (as would be measured by a clock
  ///         on the wall. It's NOT the processor time)
  std::chrono::milliseconds elapsed() const noexcept
  {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_);
  }

  bool elapsed(std::chrono::milliseconds m) const noexcept
  {
    return m > 0ms && elapsed() > m;
  }

private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace utility

///
/// Infrastructure to conduct a UCT search.
///
/// \tparam STATE
/// \parblock
/// A class that contains the present situation. The expected interface is:
///
/// ```
/// class state
/// {
/// public:
///   using action = /* the representation of an action/move */;
///                  // also required is a sentinel value `action::sentry()`
///
///   std::vector<action> actions() const;  // set of available actions in this
///                                         // state
///   void take_action(action);             // performs the required action
///                                         // changing the current state
///   unsigned agent_id() const;            // active agent
///   std::vector<double> eval() const;     // how good is this state from each
///                                         // agent's POV
///   bool is_final() const;                // returns `true` if the state is
///                                         // final
/// private:
///   // ...Hic sunt leones...
/// };
/// ```
///
/// User has to implement this class (which acts as the adapter of the adapter
/// structural design pattern).
/// \endparblock
///
/// Assumes game results in the `[0.0, 1.0]` range. Supports both single and
/// multiple agents.
///
/// \note
/// Code initially based on the Python implementation at
/// http://mcts.ai/code/index.html (Peter Cowling, Ed Powley, Daniel Whitehouse
/// - University of York).
///
template<class STATE>
class uct
{
public:
  using action   = typename STATE::action;
  using scores_t = std::invoke_result_t<decltype(&STATE::eval), STATE>;

  explicit uct(const STATE &);

  // *** FLUENT INTERFACE FOR SETTING PARAMETERS ***
  uct &exploration_bias(double) noexcept;

  uct &max_iterations(std::uintmax_t = 0) noexcept;
  uct &max_search_time(std::chrono::milliseconds = 0ms) noexcept;
  uct &simulation_depth(unsigned = 0) noexcept;

  uct &log(std::ostream &) noexcept;
  uct &log_depth(unsigned) noexcept;
  uct &verbose(bool) noexcept;

  std::pair<action, scores_t> run();

private:
  struct node;

  STATE root_state_;

  double k_;    // exploration bias parameter

  // Search time/size constraints.
  std::chrono::milliseconds max_search_time_;
  std::uintmax_t max_iterations_;
  unsigned simulation_depth_;

  // Logging related variables.
  std::ostream *log_;
  unsigned log_depth_;
  bool verbose_;
};  // class uct

///
/// A node in the game tree.
///
template<class STATE>
struct uct<STATE>::node
{
  using agent_id_t = std::invoke_result_t<decltype(&STATE::agent_id), STATE>;

  static double uct_k;

  explicit node(const STATE &, const action & = action::sentry(),
                node * = nullptr);

  node *select_child();
  node *add_child(const STATE &);
  void update(const scores_t &);

  bool fully_expanded() const;

  std::string graph(unsigned) const;

  // *** DATA MEMBERS ***
  action parent_action;  /// used during selection
  node    *parent_node;  /// used during backpropagation

  std::vector<action> untried_actions;
  std::vector<node>       child_nodes;

  /// Score of the state from multiple POVs. It's an estimated value based on
  /// simulation results.
  scores_t score;

  std::intmax_t visits;  /// number of times this node has been visited

  agent_id_t agent_id;   /// id of the active player
};  // struct uct::node

template<class STATE> double uct<STATE>::node::uct_k = 1.0;

///
/// \param[in] state
/// \param[in] parent_action the move that got us to the current node/state
///                          (`action::sentry()` for the root node)
/// \param[in] parent_node   `nullptr` for the root node
///
/// New nodes are created during the expansion phase via the `add_child`
/// method.
///
template<class STATE>
uct<STATE>::node::node(const STATE &state, const action &parent_a,
                       node *parent_n)
  : parent_action(parent_a), parent_node(parent_n),
    untried_actions(state.actions()), child_nodes(), score(), visits(0),
    agent_id(state.agent_id())
{
  child_nodes.reserve(untried_actions.capacity());  // to avoid reallocation
}

template<class STATE>
bool uct<STATE>::node::fully_expanded() const
{
  return untried_actions.empty() && child_nodes.size();
}

template<class STATE>
std::string uct<STATE>::node::graph(unsigned log_depth) const
{
  std::string ret("digraph g {");
  unsigned nodes(0);

  const std::function<void (const node &, unsigned, unsigned)> visit(
    [&](const node &n, unsigned depth, unsigned parent_id)
    {
      const auto id(++nodes);
      const auto node_name("N" + std::to_string(id));

      ret += node_name + " [label=\"" + std::to_string(n.agent_id);

      ret += " ( ";
      for (auto s : n.score)
        ret += std::to_string(s) + " ";
      ret += ")";

      ret += "/" + std::to_string(n.visits) + "\"";

      if (n.child_nodes.empty())
        ret += " shape=rectangle";

      ret += "];";

      if (parent_id)
      {
        const auto parent_name("N" + std::to_string(parent_id));
        ret += parent_name + std::string("->") + node_name;

        std::ostringstream ss;
        ss << n.parent_action;
        ret += " [label=\"" + ss.str() + "\"];";
      }

      if (depth < log_depth)
        for (const auto &child : n.child_nodes)
          visit(child, depth + 1, id);
    });

  visit(*this, 0, 0);

  ret += "}";

  return ret;
}

///
/// Selects a child node using the UCB formula.
///
template<class STATE>
typename uct<STATE>::node *uct<STATE>::node::select_child()
{
  assert(fully_expanded());  // called only during the selection phase

  const auto ucb =  // UCB score of a child node
    [this](const node &child)
    {
      if (!child.visits)
        return std::numeric_limits<typename scores_t::value_type>::max();

      assert(child.visits <= visits);
      assert(!child.score.empty());

      // Agent-just-moved point of view for the score.
      return child.score[agent_id] / child.visits
             + uct_k * std::sqrt(2 * std::log(visits) / child.visits);
    };

  return &*std::max_element(child_nodes.begin(), child_nodes.end(),
                            [ucb](const node &lhs, const node &rhs)
                            {
                              return ucb(lhs) < ucb(rhs);
                            });
}

///
/// Adds a new child to the current node and updates the list of untried
/// actions.
///
/// \param[in] s current state
/// \return      the added child node
///
/// Assumes that the child node has been reached via the last untried action.
///
template<class STATE>
typename uct<STATE>::node *uct<STATE>::node::add_child(const STATE &s)
{
  assert(untried_actions.size());

  // Reallocation would create dangling pointers.
  assert(child_nodes.size() < child_nodes.capacity());

  child_nodes.emplace_back(s, untried_actions.back(), this);
  untried_actions.pop_back();

  return &child_nodes.back();
}

///
/// Updates this node (one additional visit and additional value).
///
/// \param[in] sv score vector
///
template<class STATE>
void uct<STATE>::node::update(const scores_t &sv)
{
  ++visits;

  if (score.empty())
    score = sv;
  else
  {
    assert(score.size() == sv.size());

    std::transform(score.begin(), score.end(), sv.begin(), score.begin(),
                   std::plus());
  }
}

template<class STATE>
uct<STATE>::uct(const STATE &root_state)
  : root_state_(root_state), k_(1.0),
    max_search_time_(0ms), max_iterations_(0),
    simulation_depth_(std::numeric_limits<decltype(simulation_depth_)>::max()),
    log_(nullptr), log_depth_(1000), verbose_(false)
{
}

///
/// Sets the exploration bias parameter.
///
/// \param[in] v exploration bias term
/// \return      a reference to the current object (fluent interface)
///
/// The exploration term can be adjusted to lower or increase the amount of
/// exploration performed.
/// A value of `0` produces greedy action-selection without exploration. A very
/// high bias yields poor results since it will reduce MCTS to MC (in the
/// limit).
template<class STATE>
uct<STATE> &uct<STATE>::exploration_bias(double v) noexcept
{
  assert(v >= 0.0);

  k_ = v;
  return *this;
}

///
/// \param[in] v maximum number of iterations for the search. The default value
///              removes any constraint
/// \return      a reference to the current object (fluent interface)
///
template<class STATE>
uct<STATE> &uct<STATE>::max_iterations(std::uintmax_t v) noexcept
{
  max_iterations_ = v;
  return *this;
}

///
/// \param[in] v maximum number of milliseconds for the search. The default
///              value removes any constraint.
/// \return      a reference to the current object (fluent interface)
///
template<class STATE>
uct<STATE> &uct<STATE>::max_search_time(std::chrono::milliseconds v) noexcept
{
  max_search_time_ = v;
  return *this;
}

///
/// \param[in] v maximum simulation depth. The default value removes any
///              constraint
/// \return      a reference to the current object (fluent interface)
///
/// \warning     When setting a specific value, the `eval` function must return
///              meaningful values for NON FINAL states also.
///
template<class STATE>
uct<STATE> &uct<STATE>::simulation_depth(unsigned v) noexcept
{
  simulation_depth_ = v ? v : std::numeric_limits<decltype(v)>::max();
  return *this;
}

///
/// \param[in] s output stream for logging information
/// \return      a reference to the current object (fluent interface)
///
template<class STATE>
uct<STATE> &uct<STATE>::log(std::ostream &s) noexcept
{
  log_ = &s;
  return *this;
}

///
/// \param[in] v maximum logging depth
/// \return      a reference to the current object (fluent interface)
///
template<class STATE>
uct<STATE> &uct<STATE>::log_depth(unsigned v) noexcept
{
  log_depth_ = v;
  return *this;
}

///
/// \param[in] v enables/disables the debug output
/// \return      a reference to the current object (fluent interface)
///
template<class STATE>
uct<STATE> &uct<STATE>::verbose(bool v) noexcept
{
  verbose_ = v;
  return *this;
}

///
/// Conducts a UCT search for starting from the root state.
///
/// \return the best action from the root state
///
template<class STATE>
std::pair<typename uct<STATE>::action, typename uct<STATE>::scores_t>
uct<STATE>::run()
{
  bool stop_request(false);
  decltype(max_iterations_) iterations(0);
  utility::timer t;

  node root_node(root_state_);

  if (root_state_.is_final())
    return {uct<STATE>::action::sentry(), root_state_.eval()};

  while (!stop_request)
  {
    node *n(&root_node);
    STATE state(root_state_);

    // Selection.
    while (n->fully_expanded())
    {
      n = n->select_child();
      state.take_action(n->parent_action);
    }

    // Expansion.
    if (n->untried_actions.size())  // node can be expanded
    {
      state.take_action(n->untried_actions.back());
      n = n->add_child(state);
    }

    // Simulation (aka Playout / Rollout).
    auto remaining_depth(simulation_depth_);
    for (auto actions(n->untried_actions);
         remaining_depth && !state.is_final();
         --remaining_depth, actions = state.actions())
    {
      state.take_action(utility::random_element(actions));
    }

    // Backpropagation.
    const auto scores(state.eval());
    for (; n; n = n->parent_node)
      n->update(scores);

    // Polling for stop conditions.
    if ((++iterations & 1023) == 0)
    {
      stop_request = (max_iterations_ && iterations >= max_iterations_)
                     || t.elapsed(max_search_time_);
    }
  }

  if (verbose_)
  {
    for (const auto &n : root_node.child_nodes)
    {
      std::cout << "#-------------------------------------------\n"
                << "# move: " << n.parent_action << "   score: "
                << n.score[root_node.agent_id] << '/' << n.visits
                << std::endl;
    }
  }

  if (log_ && log_depth_)
    (*log_) << root_node.graph(log_depth_) << std::flush;

  const node &best(*std::max_element(root_node.child_nodes.begin(),
                                     root_node.child_nodes.end(),
                                     [](const node &lhs, const node &rhs)
                                     {
                                       return lhs.visits < rhs.visits;
                                     }));
  return {best.parent_action, best.score};
}

}  // namespace pocket_mcts

#endif  // include guard
