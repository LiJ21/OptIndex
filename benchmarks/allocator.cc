#include <benchmark/benchmark.h>

#include "optindex.h"

#include <boost/pool/pool_alloc.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Test object
// Keep it moderately large so allocator behavior is realistic.
// ---------------------------------------------------------------------------
struct Particle {
  uint64_t id;
  double x;
  double y;
  double m;
  double vx{}, vy{};
  double fx{}, fy{};
  double charge{};
  uint32_t flags{};
  uint32_t zone{};
  char name[32]{"a default name"};

  Particle(uint64_t id_, double x_, double y_, double m_)
      : id(id_), x(x_), y(y_), m(m_) {}
};

static constexpr size_t kN = 100'000;

// ---------------------------------------------------------------------------
// Your pool allocator wrapper benchmark target
// ---------------------------------------------------------------------------
using MyPool = optindex::FixedSizeLifoPool<Particle, kN>;

// ---------------------------------------------------------------------------
// Helper for std::allocator / boost::fast_pool_allocator
// ---------------------------------------------------------------------------
template <class Alloc>
static Particle* alloc_one(Alloc& alloc, uint64_t id) {
  Particle* p = std::allocator_traits<Alloc>::allocate(alloc, 1);
  std::allocator_traits<Alloc>::construct(alloc, p, id, 1.0, 2.0, 3.0);
  return p;
}

template <class Alloc>
static void free_one(Alloc& alloc, Particle* p) {
  std::allocator_traits<Alloc>::destroy(alloc, p);
  std::allocator_traits<Alloc>::deallocate(alloc, p, 1);
}

// ---------------------------------------------------------------------------
// 1. Allocate only until full
// Measures pure allocation + construction throughput.
// ---------------------------------------------------------------------------
static void BM_MyPool_AllocateOnly(benchmark::State& state) {
  for (auto _ : state) {
    MyPool pool;
    for (size_t i = 0; i < kN; ++i) {
      auto* p = pool.create(i, 1.0, 2.0, 3.0);
      benchmark::DoNotOptimize(p);
    }
    benchmark::DoNotOptimize(pool.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_MyPool_AllocateOnly)->Unit(benchmark::kMicrosecond);

static void BM_StdAlloc_AllocateOnly(benchmark::State& state) {
  for (auto _ : state) {
    std::allocator<Particle> alloc;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      auto* p = alloc_one(alloc, i);
      ptrs.push_back(p);
      benchmark::DoNotOptimize(p);
    }

    benchmark::DoNotOptimize(ptrs.size());

    for (auto* p : ptrs) free_one(alloc, p);
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_StdAlloc_AllocateOnly)->Unit(benchmark::kMicrosecond);

static void BM_BoostFastPool_AllocateOnly(benchmark::State& state) {
  for (auto _ : state) {
    boost::fast_pool_allocator<Particle> alloc;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      auto* p = alloc_one(alloc, i);
      ptrs.push_back(p);
      benchmark::DoNotOptimize(p);
    }

    benchmark::DoNotOptimize(ptrs.size());

    for (auto* p : ptrs) free_one(alloc, p);
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_BoostFastPool_AllocateOnly)->Unit(benchmark ::kMicrosecond);

// ---------------------------------------------------------------------------
// 2. Allocate then deallocate all
// Measures full lifecycle in FIFO destruction order.
// ---------------------------------------------------------------------------
static void BM_MyPool_AllocDeallocAll(benchmark::State& state) {
  for (auto _ : state) {
    MyPool pool;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      auto* p = pool.create(i, 1.0, 2.0, 3.0);
      ptrs.push_back(p);
    }

    for (auto* p : ptrs) {
      pool.remove(*p);
    }

    benchmark::DoNotOptimize(pool.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_MyPool_AllocDeallocAll)->Unit(benchmark::kMicrosecond);

static void BM_StdAlloc_AllocDeallocAll(benchmark::State& state) {
  for (auto _ : state) {
    std::allocator<Particle> alloc;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      ptrs.push_back(alloc_one(alloc, i));
    }

    for (auto* p : ptrs) {
      free_one(alloc, p);
    }

    benchmark::DoNotOptimize(ptrs.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_StdAlloc_AllocDeallocAll)->Unit(benchmark::kMicrosecond);

static void BM_BoostFastPool_AllocDeallocAll(benchmark::State& state) {
  for (auto _ : state) {
    boost::fast_pool_allocator<Particle> alloc;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      ptrs.push_back(alloc_one(alloc, i));
    }

    for (auto* p : ptrs) {
      free_one(alloc, p);
    }

    benchmark::DoNotOptimize(ptrs.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_BoostFastPool_AllocDeallocAll)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// 3. Allocate then deallocate in reverse order
// This is especially favorable to LIFO-style pools.
// ---------------------------------------------------------------------------
static void BM_MyPool_AllocDeallocReverse(benchmark::State& state) {
  for (auto _ : state) {
    MyPool pool;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      ptrs.push_back(pool.create(i, 1.0, 2.0, 3.0));
    }

    for (size_t i = ptrs.size(); i > 0; --i) {
      pool.remove(*ptrs[i - 1]);
    }

    benchmark::DoNotOptimize(pool.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_MyPool_AllocDeallocReverse)->Unit(benchmark::kMicrosecond);

static void BM_StdAlloc_AllocDeallocReverse(benchmark::State& state) {
  for (auto _ : state) {
    std::allocator<Particle> alloc;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      ptrs.push_back(alloc_one(alloc, i));
    }

    for (size_t i = ptrs.size(); i > 0; --i) {
      free_one(alloc, ptrs[i - 1]);
    }

    benchmark::DoNotOptimize(ptrs.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_StdAlloc_AllocDeallocReverse)->Unit(benchmark::kMicrosecond);

static void BM_BoostFastPool_AllocDeallocReverse(benchmark::State& state) {
  for (auto _ : state) {
    boost::fast_pool_allocator<Particle> alloc;
    std::vector<Particle*> ptrs;
    ptrs.reserve(kN);

    for (size_t i = 0; i < kN; ++i) {
      ptrs.push_back(alloc_one(alloc, i));
    }

    for (size_t i = ptrs.size(); i > 0; --i) {
      free_one(alloc, ptrs[i - 1]);
    }

    benchmark::DoNotOptimize(ptrs.size());
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_BoostFastPool_AllocDeallocReverse)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// 4. Steady-state churn
// Keep a fixed live population and repeatedly free one / allocate one.
// This is the most realistic allocator benchmark for container churn.
// ---------------------------------------------------------------------------
static void BM_MyPool_Churn(benchmark::State& state) {
  constexpr size_t kLive = 50'000;
  constexpr size_t kOps = 200'000;

  for (auto _ : state) {
    MyPool pool;
    std::vector<Particle*> live;
    live.reserve(kLive);

    for (size_t i = 0; i < kLive; ++i) {
      live.push_back(pool.create(i, 1.0, 2.0, 3.0));
    }

    size_t next_id = kLive;
    size_t pos = 0;

    for (size_t op = 0; op < kOps; ++op) {
      pool.remove(*live[pos]);
      live[pos] = pool.create(next_id++, 1.0, 2.0, 3.0);
      pos++;
      if (pos == kLive) pos = 0;
    }

    benchmark::DoNotOptimize(pool.size());
  }

  state.SetItemsProcessed(state.iterations() * 200'000);
}
BENCHMARK(BM_MyPool_Churn)->Unit(benchmark::kMicrosecond);

static void BM_StdAlloc_Churn(benchmark::State& state) {
  constexpr size_t kLive = 50'000;
  constexpr size_t kOps = 200'000;

  for (auto _ : state) {
    std::allocator<Particle> alloc;
    std::vector<Particle*> live;
    live.reserve(kLive);

    for (size_t i = 0; i < kLive; ++i) {
      live.push_back(alloc_one(alloc, i));
    }

    size_t next_id = kLive;
    size_t pos = 0;

    for (size_t op = 0; op < kOps; ++op) {
      free_one(alloc, live[pos]);
      live[pos] = alloc_one(alloc, next_id++);
      pos++;
      if (pos == kLive) pos = 0;
    }

    for (auto* p : live) free_one(alloc, p);

    benchmark::DoNotOptimize(live.size());
  }

  state.SetItemsProcessed(state.iterations() * 200'000);
}
BENCHMARK(BM_StdAlloc_Churn)->Unit(benchmark::kMicrosecond);

static void BM_BoostFastPool_Churn(benchmark::State& state) {
  constexpr size_t kLive = 50'000;
  constexpr size_t kOps = 200'000;

  for (auto _ : state) {
    boost::fast_pool_allocator<Particle> alloc;
    std::vector<Particle*> live;
    live.reserve(kLive);

    for (size_t i = 0; i < kLive; ++i) {
      live.push_back(alloc_one(alloc, i));
    }

    size_t next_id = kLive;
    size_t pos = 0;

    for (size_t op = 0; op < kOps; ++op) {
      free_one(alloc, live[pos]);
      live[pos] = alloc_one(alloc, next_id++);
      pos++;
      if (pos == kLive) pos = 0;
    }

    for (auto* p : live) free_one(alloc, p);

    benchmark::DoNotOptimize(live.size());
  }

  state.SetItemsProcessed(state.iterations() * 200'000);
}
BENCHMARK(BM_BoostFastPool_Churn)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();