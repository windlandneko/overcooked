// Vectorized Overcooked under the course rules (core/overcooked_mdp.py with one order), driven by engine.py.
// Each reset and step writes observations, timesteps, rewards, done flags and events into buffers the
// caller binds at creation.

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

namespace {

// Codes of engine.TERRAIN; a non-floor tile lights observation channel 9 + code.
enum Terrain : uint8_t { FLOOR, POT, COUNTER, ONION_BOX, TOMATO_BOX, DISH_BOX, SERVE };
enum Kind : uint8_t { NONE, ONION, TOMATO, DISH, SOUP };
// Per-seat event channels, in the order of engine.EVENTS.
enum Event {
  POT_ONION, POT_TOMATO, START_COOK, DISH_PICKUP, SOUP_PICKUP, SOUP_DELIVER,
  DROP_ONION, HANDOFF_PICKUP, DROP_SOUP, COLLISION, CURL, ZONE, NUM_EVENTS,
};
// Channels of one seat's view: 0 self, 1 partner, 2-5 own facing, 6-9 partner facing, 10-15 terrain, then these.
enum Channel {
  IDLE_ONIONS = 16, IDLE_TOMATOES, SOUP_ONIONS, SOUP_TOMATOES, COOK_LEFT, SOUP_READY,
  DISH_ITEM, ONION_ITEM, TOMATO_ITEM, URGENT, CHANNELS,
};

constexpr std::array DX{0, 0, 1, -1};  // north, south, east, west; action 4 stays, 5 interacts
constexpr std::array DY{-1, 1, 0, 0};
constexpr int INTERACT = 5;
constexpr int RING_CAPACITY = 8192;
constexpr int URGENT_STEPS = 40;

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

struct Rng {
  uint64_t state;

  uint64_t next() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
  }
  int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
  int pick(const auto& values) { return values[below(static_cast<int>(values.size()))]; }
  float uniform() { return static_cast<float>(next() >> 40) * 0x1p-24f; }
};

// The set bits of a mask, lowest first.
struct Bits {
  uint64_t mask;

  struct Iterator {
    uint64_t rest;
    int operator*() const { return std::countr_zero(rest); }
    Iterator& operator++() { rest &= rest - 1; return *this; }
    bool operator!=(const Iterator& other) const { return rest != other.rest; }
  };
  [[nodiscard]] Iterator begin() const { return {mask}; }
  [[nodiscard]] Iterator end() const { return {0}; }
};

struct Item {
  int32_t ready = -1;  // soups: timestep the soup is done, -1 while an idle pot waits to be started
  Kind kind = NONE;
  uint8_t onions = 0;
  uint8_t tomatoes = 0;
  int8_t holder = -1;        // chef who last held it, -1 when fresh from a dispenser or pot
  bool handed_over = false;  // the teammate pickup event fires once per item

  [[nodiscard]] int ingredients() const { return onions + tomatoes; }
};

struct State {
  uint64_t occupied = 0;  // one bit per slot: counters first, then pots
  std::array<int, 2> pos{};
  std::array<uint8_t, 2> facing{};
  std::array<Item, 2> held{};
  int timestep = 0;
  bool augmented = false;
};

struct Move {
  int pos;
  float curl;  // movement around the kitchen centre, +1 clockwise
};

struct Target {
  Terrain terrain = FLOOR;
  int slot = -1;
  float zone = 0;  // +1 in the lower or right half of the kitchen, -1 otherwise
};

// Both seat views of one environment's observation.
struct Grid {
  float* data;
  int cells;

  void set(int channel, int pos, float value) const {
    data[channel * cells + pos] = value;
    data[(CHANNELS + channel) * cells + pos] = value;
  }

  void chef(int seat, int pos, int facing) const {
    for (int view : {0, 1}) {
      const int role = seat ^ view;  // 0 in the chef's own view, 1 in its partner's
      float* grid = data + view * CHANNELS * cells;
      grid[role * cells + pos] = 1;
      grid[(2 + 4 * role + facing) * cells + pos] = 1;
    }
  }

  void item(const Item& item, int pos, int timestep) const {
    switch (item.kind) {
      case DISH: set(DISH_ITEM, pos, 1); break;
      case ONION: set(ONION_ITEM, pos, 1); break;
      case TOMATO: set(TOMATO_ITEM, pos, 1); break;
      case SOUP: {
        const bool idle = item.ready < 0;
        set(idle ? IDLE_ONIONS : SOUP_ONIONS, pos, item.onions);
        set(idle ? IDLE_TOMATOES : SOUP_TOMATOES, pos, item.tomatoes);
        if (idle) break;
        if (item.ready > timestep) set(COOK_LEFT, pos, static_cast<float>(item.ready - timestep));
        else set(SOUP_READY, pos, 1);
        break;
      }
      case NONE: break;
    }
  }
};

// Calls fn(e) for every environment, splitting large batches into contiguous blocks across threads.
void parallel_for(int count, const auto& fn) {
  constexpr int GRAIN = 256;
#ifdef __APPLE__
  const int blocks = std::min(6, count / GRAIN);
  if (blocks > 1) {
    struct Job {
      decltype(fn) fn;
      int count, blocks;
    } job{fn, count, blocks};
    dispatch_apply_f(blocks, DISPATCH_APPLY_AUTO, &job, [](void* context, size_t block) {
      const auto& [fn, count, blocks] = *static_cast<const Job*>(context);
      const int end = static_cast<int>(count * (block + 1) / blocks);
      for (int e = static_cast<int>(count * block / blocks); e < end; ++e) fn(e);
    });
    return;
  }
#endif
#pragma omp parallel for schedule(static) if (count >= 2 * GRAIN)
  for (int e = 0; e < count; ++e) fn(e);
}

struct Outputs {
  float* obs;          // (count, 2, CHANNELS, cells)
  int32_t* timesteps;  // (count,)
  float* sparse;       // (count,)
  float* shaped;       // (count,)
  bool* done;          // (count,)
  float* events;       // (count, 2, NUM_EVENTS)
  bool* augmented;     // (count,) the running game began mid-game
};

struct VecEnv {
  int count, cells, num_slots;
  std::array<int, 2> start;
  int onions, tomatoes, cook_time, reward, horizon;
  bool has_tomatoes = false;
  uint64_t pot_mask = 0;

  std::vector<std::array<Move, 4>> moves;
  std::vector<std::array<Target, 4>> targets;
  std::vector<int> slot_pos;                     // counters, then pots
  std::vector<int> reachable_counters;           // counters next to a floor tile either chef can reach
  std::array<std::vector<int>, 2> floors;        // floor tiles each chef can reach
  std::vector<std::array<int, 3>> facing_pairs;  // seat-0 tile, adjacent seat-1 tile, direction 0 -> 1
  std::vector<float> empty_grid;                 // terrain only, both views

  // Live environments, then the replay ring, sharing one slot layout.
  std::vector<State> states;
  std::vector<Item> slots;
  std::vector<Rng> rngs;
  Rng rng{};
  int ring_next = 0;
  int ring_size = 0;

  Outputs out;

  std::span<Item> items(int e) { return {slots.data() + static_cast<size_t>(e) * num_slots, static_cast<size_t>(num_slots)}; }

  VecEnv(int count, int width, int height, const uint8_t* terrain, int start0, int start1,
         int onions, int tomatoes, int cook_time, int reward, int horizon, Outputs out)
      : count(count), cells(width * height), start{start0, start1},
        onions(onions), tomatoes(tomatoes), cook_time(cook_time), reward(reward), horizon(horizon),
        moves(cells), targets(cells), empty_grid(2 * CHANNELS * cells),
        states(count + RING_CAPACITY), rngs(count), out(out) {
    std::vector<int> slot_of(cells, -1);
    std::vector<int> pots;
    for (int pos = 0; pos < cells; ++pos) {
      if (terrain[pos] == FLOOR) continue;
      Grid{empty_grid.data(), cells}.set(9 + terrain[pos], pos, 1);
      if (terrain[pos] == COUNTER) {
        slot_of[pos] = static_cast<int>(slot_pos.size());
        slot_pos.push_back(pos);
      }
      if (terrain[pos] == POT) pots.push_back(pos);
      if (terrain[pos] == TOMATO_BOX) has_tomatoes = true;
    }
    const int num_counters = static_cast<int>(slot_pos.size());
    for (int pos : pots) {
      slot_of[pos] = static_cast<int>(slot_pos.size());
      pot_mask |= uint64_t{1} << slot_of[pos];
      slot_pos.push_back(pos);
    }
    num_slots = static_cast<int>(slot_pos.size());
    slots.resize(states.size() * num_slots);

    const float cx = 0.5f * (width - 1);
    const float cy = 0.5f * (height - 1);
    std::vector<std::array<int, 4>> neighbours(cells);
    for (int pos = 0; pos < cells; ++pos) {
      const int x = pos % width;
      const int y = pos / width;
      for (int d = 0; d < 4; ++d) {
        const int nx = x + DX[d];
        const int ny = y + DY[d];
        const bool inside = nx >= 0 && nx < width && ny >= 0 && ny < height;
        const int next = ny * width + nx;
        neighbours[pos][d] = inside ? next : -1;
        moves[pos][d] = {pos, 0};
        if (!inside) continue;
        targets[pos][d] = {static_cast<Terrain>(terrain[next]), slot_of[next],
                           ny > cy || (ny == cy && nx > cx) ? 1.0f : -1.0f};
        if (terrain[next] == FLOOR) {
          moves[pos][d] = {next, std::clamp((cy - y) * DX[d] + (x - cx) * DY[d], -1.0f, 1.0f)};
        }
      }
    }

    std::array<std::vector<bool>, 2> reached;
    for (int seat : {0, 1}) {
      reached[seat].assign(cells, false);
      reached[seat][start[seat]] = true;
      floors[seat] = {start[seat]};
      for (size_t i = 0; i < floors[seat].size(); ++i) {
        for (int next : neighbours[floors[seat][i]]) {
          if (next >= 0 && terrain[next] == FLOOR && !reached[seat][next]) {
            reached[seat][next] = true;
            floors[seat].push_back(next);
          }
        }
      }
    }
    for (int pos : floors[0]) {
      for (int d = 0; d < 4; ++d) {
        const int next = neighbours[pos][d];
        if (next >= 0 && reached[1][next]) facing_pairs.push_back({pos, next, d});
      }
    }
    for (int slot = 0; slot < num_counters; ++slot) {
      if (std::ranges::any_of(neighbours[slot_pos[slot]],
                              [&](int next) { return next >= 0 && (reached[0][next] || reached[1][next]); })) {
        reachable_counters.push_back(slot);
      }
    }

    seed(42);
    parallel_for(count, [&](int e) {
      restart(e);
      encode(e);
    });
  }

  void seed(uint64_t value) {
    rng.state = splitmix64(value);
    for (int e = 0; e < count; ++e) rngs[e].state = splitmix64(value + 0x9e3779b97f4a7c15ULL * (e + 1));
  }

  [[nodiscard]] Item order_soup() const {
    return {.ready = 0, .kind = SOUP, .onions = static_cast<uint8_t>(onions), .tomatoes = static_cast<uint8_t>(tomatoes)};
  }

  void restart(int e) { states[e] = {.pos = start}; }

  void copy(int to, int from) {
    states[to] = states[from];
    std::ranges::copy(items(from), items(to).begin());
  }

  // A plausible mid-game kitchen: chefs on reachable tiles holding random items, clutter on reachable
  // counters, and pots that are empty, filling, cooking, ready or holding a wrong recipe.
  void randomize(int e) {
    State& s = states[e];
    Rng& r = rngs[e];
    const std::span<Item> board = items(e);
    s = {.augmented = true};

    if (!facing_pairs.empty() && r.uniform() < 0.20f) {
      const auto [p0, p1, d] = facing_pairs[r.below(static_cast<int>(facing_pairs.size()))];
      s.pos = {p0, p1};
      s.facing = {static_cast<uint8_t>(d), static_cast<uint8_t>(d ^ 1)};  // face each other
    } else {
      const auto& f1 = floors[1];
      s.pos = {r.pick(floors[0]), r.pick(f1)};
      if (s.pos[1] == s.pos[0] && f1.size() > 1) {
        const int k = r.below(static_cast<int>(f1.size()) - 1);
        s.pos[1] = f1[k] == s.pos[0] ? f1.back() : f1[k];
      }
      s.facing[0] = static_cast<uint8_t>(r.below(4));
      s.facing[1] = static_cast<uint8_t>(r.below(4));
    }

    for (Item& held : s.held) {
      const float u = r.uniform();
      held = u < 0.45f ? Item{} : u < 0.75f ? Item{.kind = ONION} : u < 0.90f ? Item{.kind = DISH} : order_soup();
    }

    for (int slot : reachable_counters) {
      if (r.uniform() >= 0.18f) continue;
      const float u = r.uniform();
      board[slot] = u < 0.60f ? Item{.kind = ONION} : u < 0.80f ? Item{.kind = DISH} : order_soup();
      s.occupied |= uint64_t{1} << slot;
    }

    s.timestep = r.below(std::max(1, horizon / 2));
    for (int slot : Bits{pot_mask}) {
      const float u = r.uniform();
      if (u < 0.25f) continue;
      Item soup = order_soup();
      int left = -1;  // cook steps left, -1 for an idle pot
      if (u < 0.55f) {
        soup = {.kind = SOUP, .onions = static_cast<uint8_t>(1 + r.below(3))};
      } else if (u < 0.80f) {
        left = cook_time - r.below(cook_time);
      } else if (u < 0.90f) {
        left = 0;
      } else {  // a cooking or ready soup that does not match the order
        left = r.uniform() < 0.5f ? 0 : cook_time - r.below(cook_time);
        if (has_tomatoes && tomatoes == 0 && r.uniform() < 0.5f) {
          soup.onions = static_cast<uint8_t>(r.below(3));
          soup.tomatoes = 1;
        } else {
          soup.onions = static_cast<uint8_t>(1 + r.below(2));
          soup.tomatoes = 0;
        }
      }
      soup.ready = left < 0 ? -1 : s.timestep + left;
      board[slot] = soup;
      s.occupied |= uint64_t{1} << slot;
    }
  }

  void reset(int e, float aug_prob) {
    Rng& r = rngs[e];
    if (aug_prob <= 0.0f || r.uniform() >= aug_prob) {
      restart(e);
    } else if (ring_size >= 256 && r.uniform() < 0.70f) {
      copy(e, count + r.below(ring_size));
      states[e].augmented = true;
    } else {
      randomize(e);
    }
  }

  void record(const bool* mask, int stride) {
    for (int e = rng.below(stride); e < count; e += stride) {
      const State& s = states[e];
      if (!mask[e] || s.augmented || s.timestep < 10 || s.timestep >= horizon * 3 / 4) continue;
      copy(count + ring_next, e);
      ring_next = (ring_next + 1) % RING_CAPACITY;
      ring_size = std::min(ring_size + 1, RING_CAPACITY);
    }
  }

  // Pots a dish could serve: partially filled, cooking or ready. A full pot not yet started does not count.
  [[nodiscard]] int pots_needing_dishes(const State& s, std::span<const Item> board) const {
    int n = 0;
    for (int slot : Bits{s.occupied & pot_mask}) n += board[slot].ready >= 0 || board[slot].ingredients() < 3;
    return n;
  }

  [[nodiscard]] bool dish_on_counter(const State& s, std::span<const Item> board) const {
    for (int slot : Bits{s.occupied & ~pot_mask}) {
      if (board[slot].kind == DISH) return true;
    }
    return false;
  }

  void interact(State& s, std::span<Item> board, int p, int dish_pots, float* ev, float& sparse, float& shaped) {
    const Target& target = targets[s.pos[p]][s.facing[p]];
    const uint64_t bit = target.slot >= 0 ? uint64_t{1} << target.slot : 0;
    const bool filled = s.occupied & bit;
    Item& held = s.held[p];
    switch (target.terrain) {
      case COUNTER: {
        Item& top = board[target.slot];
        if (filled && held.kind == NONE) {
          held = top;
          if (held.holder == 1 - p && !held.handed_over) {
            ev[HANDOFF_PICKUP] = 1;
            held.handed_over = true;
          }
          held.holder = static_cast<int8_t>(p);
          s.occupied ^= bit;
        } else if (!filled && held.kind != NONE) {
          if (held.holder < 0) {
            ev[DROP_ONION] = held.kind == ONION;
            ev[DROP_SOUP] = held.kind == SOUP;
          }
          top = held;
          top.holder = static_cast<int8_t>(p);
          held = {};
          s.occupied ^= bit;
        }
        break;
      }
      case ONION_BOX:
        if (held.kind != NONE) break;
        held = {.kind = ONION};
        ev[ZONE] = target.zone;
        break;
      case TOMATO_BOX:
        if (held.kind == NONE) held = {.kind = TOMATO};
        break;
      case DISH_BOX:
        if (held.kind != NONE) break;
        if (int{s.held[1 - p].kind == DISH} < dish_pots && !dish_on_counter(s, board)) {
          shaped += 3;
          ev[DISH_PICKUP] = 1;
        }
        held = {.kind = DISH};
        ev[ZONE] = target.zone;
        break;
      case POT: {
        Item& soup = board[target.slot];
        if (held.kind == NONE && filled && soup.ready < 0) {
          soup.ready = s.timestep + cook_time;
          ev[START_COOK] = soup.onions == onions && soup.tomatoes == tomatoes;
          ev[ZONE] = target.zone;
        } else if (held.kind == DISH && filled && soup.ready >= 0 && soup.ready <= s.timestep) {
          held = soup;
          s.occupied ^= bit;
          shaped += 5;
          ev[SOUP_PICKUP] = 1;
          ev[ZONE] = target.zone;
        } else if ((held.kind == ONION || held.kind == TOMATO) &&
                   (!filled || (soup.ready < 0 && soup.ingredients() < 3))) {
          if (!filled) soup = {.kind = SOUP};
          if (held.kind == ONION) {
            ++soup.onions;
            ev[POT_ONION] = 1;
          } else {
            ++soup.tomatoes;
            ev[POT_TOMATO] = 1;
          }
          s.occupied |= bit;
          held = {};
          shaped += 3;
          ev[ZONE] = target.zone;
        }
        break;
      }
      case SERVE:
        if (held.kind != SOUP) break;
        if (held.onions == onions && held.tomatoes == tomatoes) {
          sparse += reward;
          ev[SOUP_DELIVER] = 1;
        }
        held = {};
        ev[ZONE] = target.zone;
        break;
      case FLOOR: break;
    }
  }

  // Interacts resolve first, seat 0 before seat 1, then both chefs move unless they would collide.
  bool step(int e, const int32_t* action, float& sparse, float& shaped, float* ev) {
    State& s = states[e];
    const std::span<Item> board = items(e);
    sparse = shaped = 0;
    std::fill_n(ev, 2 * NUM_EVENTS, 0.0f);

    if (action[0] == INTERACT || action[1] == INTERACT) {
      const int dish_pots = pots_needing_dishes(s, board);  // counted before either interact
      for (int p : {0, 1}) {
        if (action[p] == INTERACT) interact(s, board, p, dish_pots, ev + p * NUM_EVENTS, sparse, shaped);
      }
    }

    std::array<Move, 2> move;
    for (int p : {0, 1}) {
      move[p] = {s.pos[p], 0};
      if (action[p] < 4) {
        s.facing[p] = static_cast<uint8_t>(action[p]);
        move[p] = moves[s.pos[p]][action[p]];
      }
    }
    if (move[0].pos == move[1].pos || (move[0].pos == s.pos[1] && move[1].pos == s.pos[0])) {
      for (int p : {0, 1}) ev[p * NUM_EVENTS + COLLISION] = move[p].pos != s.pos[p];
    } else {
      for (int p : {0, 1}) {
        s.pos[p] = move[p].pos;
        ev[p * NUM_EVENTS + CURL] = move[p].curl;
      }
    }
    return ++s.timestep >= horizon;
  }

  void encode(int e) {
    const State& s = states[e];
    const Grid grid{out.obs + static_cast<size_t>(e) * empty_grid.size(), cells};
    std::ranges::copy(empty_grid, grid.data);
    for (int p : {0, 1}) {
      grid.chef(p, s.pos[p], s.facing[p]);
      grid.item(s.held[p], s.pos[p], s.timestep);
    }
    for (int slot : Bits{s.occupied}) grid.item(items(e)[slot], slot_pos[slot], s.timestep);
    if (horizon - s.timestep < URGENT_STEPS) {
      std::fill_n(grid.data + URGENT * cells, cells, 1.0f);
      std::fill_n(grid.data + (CHANNELS + URGENT) * cells, cells, 1.0f);
    }
    out.timesteps[e] = s.timestep;
    out.augmented[e] = s.augmented;
  }
};

}  // namespace

extern "C" {

void* env_create(int count, int width, int height, const uint8_t* terrain, int start0, int start1,
                 int onions, int tomatoes, int cook_time, int reward, int horizon,
                 float* obs, int32_t* timesteps, float* sparse, float* shaped, bool* done, float* events,
                 bool* augmented) {
  return new VecEnv(count, width, height, terrain, start0, start1, onions, tomatoes, cook_time, reward,
                    horizon, {obs, timesteps, sparse, shaped, done, events, augmented});
}

void env_destroy(void* env) { delete static_cast<VecEnv*>(env); }

void env_seed(void* env, uint64_t seed) { static_cast<VecEnv*>(env)->seed(seed); }

void env_reset(void* env, const bool* mask, float aug_prob) {
  auto& v = *static_cast<VecEnv*>(env);
  parallel_for(v.count, [&](int e) {
    if (!mask[e]) return;
    v.reset(e, aug_prob);
    v.encode(e);
  });
}

void env_step(void* env, const int32_t* actions, float aug_prob) {
  auto& v = *static_cast<VecEnv*>(env);
  const Outputs& out = v.out;
  parallel_for(v.count, [&](int e) {
    out.done[e] = v.step(e, actions + 2 * e, out.sparse[e], out.shaped[e], out.events + e * 2 * NUM_EVENTS);
    if (out.done[e]) v.reset(e, aug_prob);
    v.encode(e);
  });
}

void env_record(void* env, const bool* mask, int stride) { static_cast<VecEnv*>(env)->record(mask, stride); }

int env_ring_size(void* env) { return static_cast<VecEnv*>(env)->ring_size; }

}  // extern "C"
