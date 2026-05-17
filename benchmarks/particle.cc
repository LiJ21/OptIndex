#include <benchmark/benchmark.h>

#include "optindex.h"

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace bmi = boost::multi_index;

// ---------------------------------------------------------------------------
// Domain object
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

  Particle(uint64_t id, double x, double y, double m)
      : id(id), x(x), y(y), m(m) {}
};

struct IdHash {
  size_t operator()(uint64_t id) const { return std::hash<uint64_t>{}(id); }
};
struct IdEqual {
  bool operator()(uint64_t a, uint64_t b) const { return a == b; }
};

#ifdef SET_KN
static constexpr size_t kN = SET_KN;
#else
static constexpr size_t kN = 100'000;
#endif
static constexpr size_t kBuckets = kN;
// static constexpr size_t kBuckets = 131072;

static constexpr size_t mixed_steady_batch_size() {
  size_t batch = kN / 100;
  if (batch < 8)
    batch = 8;
  if (batch > 1024)
    batch = 1024;
  if (batch > kN)
    batch = kN;
  return batch;
}

static constexpr size_t kMixedSteadyRounds = 4;
static constexpr size_t kMixedSteadyBatch = mixed_steady_batch_size();

struct ById {};
struct ByX {};
struct ByY {};
struct ByM {};
struct BySeq {};

// Mass-bucket tags for the partitioned variant. Test data rounds mass to
// integers in {1..10}, so K=10 — one bucket per integer.
struct BktM1 {};
struct BktM2 {};
struct BktM3 {};
struct BktM4 {};
struct BktM5 {};
struct BktM6 {};
struct BktM7 {};
struct BktM8 {};
struct BktM9 {};
struct BktM10 {};

// ---------------------------------------------------------------------------
// Primary key configuration (shared by OptIndex/PartOptIndex/BMI/PoolBMI)
// ---------------------------------------------------------------------------
#if defined(WITH_HASH) && WITH_HASH
using ParticlePrimaryIndex =
    optindex::Unordered<optindex::KeyFrom<&Particle::id>, IdHash, IdEqual,
                        kBuckets>;
using BMIPriIndex =
    bmi::hashed_unique<bmi::tag<ById>,
                       bmi::member<Particle, uint64_t, &Particle::id>>;
#else
using ParticlePrimaryIndex =
    optindex::Ordered<optindex::KeyFrom<&Particle::id>, std::less<uint64_t>>;
using BMIPriIndex =
    bmi::ordered_unique<bmi::tag<ById>,
                        bmi::member<Particle, uint64_t, &Particle::id>>;
#endif

// ---------------------------------------------------------------------------
// OptIndex (single ByM ordered_non_unique mass index)
// ---------------------------------------------------------------------------
using ParticleMap = optindex::FixedSizeOptIndex<
    Particle, kN, ParticlePrimaryIndex,
#if defined(REV_INDEX) && REV_INDEX
    optindex::IndexBy<optindex::List, BySeq>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::x>, std::less<double>>,
        ByX>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::y>, std::less<double>>,
        ByY>,
    optindex::IndexBy<optindex::OrderedNonUnique<
                          optindex::KeyFrom<&Particle::m>, std::less<double>>,
                      ByM>
#else
    optindex::IndexBy<optindex::List, BySeq>,
    optindex::IndexBy<optindex::OrderedNonUnique<
                          optindex::KeyFrom<&Particle::m>, std::less<double>>,
                      ByM>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::y>, std::less<double>>,
        ByY>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::x>, std::less<double>>,
        ByX>
#endif
    >;

// ---------------------------------------------------------------------------
// Partitioned OptIndex: ByM replaced by K=10 List buckets, one per integer mass
// ---------------------------------------------------------------------------
using PartParticleMap = optindex::FixedSizeOptIndex<
    Particle, kN, ParticlePrimaryIndex,
#if defined(REV_INDEX) && REV_INDEX
    optindex::IndexBy<optindex::List, BySeq>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::x>, std::less<double>>,
        ByX>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::y>, std::less<double>>,
        ByY>,
    optindex::IndexBy<optindex::List, BktM1, BktM2, BktM3, BktM4, BktM5, BktM6,
                      BktM7, BktM8, BktM9, BktM10>
#else
    optindex::IndexBy<optindex::List, BySeq>,
    optindex::IndexBy<optindex::List, BktM1, BktM2, BktM3, BktM4, BktM5, BktM6,
                      BktM7, BktM8, BktM9, BktM10>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::y>, std::less<double>>,
        ByY>,
    optindex::IndexBy<
        optindex::Ordered<optindex::KeyFrom<&Particle::x>, std::less<double>>,
        ByX>
#endif
    >;

// ---------------------------------------------------------------------------
// Boost.MultiIndex (default + pool allocator)
// ---------------------------------------------------------------------------
using ParticleBMIAlloc =
    boost::fast_pool_allocator<Particle,
                               boost::default_user_allocator_new_delete,
                               boost::details::pool::null_mutex, kN>;

using BMIIndexedBy = bmi::indexed_by<
    BMIPriIndex,
    bmi::ordered_unique<bmi::tag<ByX>,
                        bmi::member<Particle, double, &Particle::x>>,
    bmi::ordered_unique<bmi::tag<ByY>,
                        bmi::member<Particle, double, &Particle::y>>,
    bmi::ordered_non_unique<bmi::tag<ByM>,
                            bmi::member<Particle, double, &Particle::m>>,
    bmi::sequenced<bmi::tag<BySeq>>>;

using ParticleBMI = bmi::multi_index_container<Particle, BMIIndexedBy>;
using ParticleBMIPool =
    bmi::multi_index_container<Particle, BMIIndexedBy, ParticleBMIAlloc>;

template <typename Map> static void reserve_primary_hash_if_enabled(Map &m) {
#if defined(WITH_HASH) && WITH_HASH
  m.reserve(kN);
#else
  (void)m;
#endif
}

template <typename Map> static void rehash_primary_if_enabled(Map &m) {
#if defined(WITH_HASH) && WITH_HASH
  m.template get<ById>().rehash(kN);
#else
  (void)m;
#endif
}

template <typename Map> static void prepare_bmi(Map &m) {
  reserve_primary_hash_if_enabled(m);
}

template <typename Map> static void prepare_pool_bmi(Map &m) {
  reserve_primary_hash_if_enabled(m);
  rehash_primary_if_enabled(m);
}

// ---------------------------------------------------------------------------
// Bucket dispatch helpers for the partitioned variant
// ---------------------------------------------------------------------------
inline int mass_to_bucket(double m) {
  int b = static_cast<int>(std::lround(m)) - 1;
  if (b < 0)
    b = 0;
  if (b > 9)
    b = 9;
  return b;
}

template <class Slot>
static inline void part_insert_bucket(PartParticleMap &m, const Slot &s,
                                      int bkt) {
  switch (bkt) {
  case 0: m.template insert<BktM1>(s); break;
  case 1: m.template insert<BktM2>(s); break;
  case 2: m.template insert<BktM3>(s); break;
  case 3: m.template insert<BktM4>(s); break;
  case 4: m.template insert<BktM5>(s); break;
  case 5: m.template insert<BktM6>(s); break;
  case 6: m.template insert<BktM7>(s); break;
  case 7: m.template insert<BktM8>(s); break;
  case 8: m.template insert<BktM9>(s); break;
  default: m.template insert<BktM10>(s); break;
  }
}

template <class Slot>
static inline void part_unindex_bucket(PartParticleMap &m, const Slot &s,
                                       int bkt) {
  switch (bkt) {
  case 0: m.template unindex<BktM1>(s); break;
  case 1: m.template unindex<BktM2>(s); break;
  case 2: m.template unindex<BktM3>(s); break;
  case 3: m.template unindex<BktM4>(s); break;
  case 4: m.template unindex<BktM5>(s); break;
  case 5: m.template unindex<BktM6>(s); break;
  case 6: m.template unindex<BktM7>(s); break;
  case 7: m.template unindex<BktM8>(s); break;
  case 8: m.template unindex<BktM9>(s); break;
  default: m.template unindex<BktM10>(s); break;
  }
}

static inline size_t part_count_bucket(PartParticleMap &m, int bkt) {
  size_t c = 0;
  switch (bkt) {
  case 0: for (auto &x : m.get<BktM1>())  { (void)x; ++c; } break;
  case 1: for (auto &x : m.get<BktM2>())  { (void)x; ++c; } break;
  case 2: for (auto &x : m.get<BktM3>())  { (void)x; ++c; } break;
  case 3: for (auto &x : m.get<BktM4>())  { (void)x; ++c; } break;
  case 4: for (auto &x : m.get<BktM5>())  { (void)x; ++c; } break;
  case 5: for (auto &x : m.get<BktM6>())  { (void)x; ++c; } break;
  case 6: for (auto &x : m.get<BktM7>())  { (void)x; ++c; } break;
  case 7: for (auto &x : m.get<BktM8>())  { (void)x; ++c; } break;
  case 8: for (auto &x : m.get<BktM9>())  { (void)x; ++c; } break;
  default: for (auto &x : m.get<BktM10>()) { (void)x; ++c; } break;
  }
  return c;
}

// ---------------------------------------------------------------------------
// Shared test data
// ---------------------------------------------------------------------------
struct TestData {
  std::vector<uint64_t> ids;
  std::vector<double> xs, ys, ms;
  std::vector<uint64_t> lookup_ids;
  std::vector<double> lookup_ms;

  TestData() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> coord(-1000.0, 1000.0);
    std::uniform_real_distribution<double> mass(1.0, 10.0);

    ids.resize(kN);
    xs.resize(kN);
    ys.resize(kN);
    ms.resize(kN);
    for (size_t i = 0; i < kN; ++i) {
      ids[i] = i + 1;
      xs[i] = coord(rng);
      ys[i] = coord(rng);
      ms[i] = std::round(mass(rng));
    }

    lookup_ids.resize(256);
    std::uniform_int_distribution<size_t> pick(0, kN - 1);
    for (auto &id : lookup_ids)
      id = ids[pick(rng)];

    lookup_ms.resize(64);
    for (auto &mv : lookup_ms)
      mv = std::round(mass(rng));
  }
};

static const TestData kData;

// ---------------------------------------------------------------------------
// Populators
// ---------------------------------------------------------------------------
static auto make_optindex() {
  auto m = std::make_unique<ParticleMap>();
  for (size_t i = 0; i < kN; ++i)
    m->create_all(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
  return m;
}

static auto make_bmi() {
  auto m = std::make_unique<ParticleBMI>();
  prepare_bmi(*m);
  for (size_t i = 0; i < kN; ++i)
    m->insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
  return m;
}

static auto make_bmi_pool() {
  auto m = std::make_unique<ParticleBMIPool>();
  prepare_pool_bmi(*m);
  for (size_t i = 0; i < kN; ++i)
    m->insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
  return m;
}

static auto make_part_optindex() {
  auto m = std::make_unique<PartParticleMap>();
  for (size_t i = 0; i < kN; ++i) {
    auto it = m->create(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
    m->insert<BySeq>(*it);
    m->insert<ByX>(*it);
    m->insert<ByY>(*it);
    part_insert_bucket(*m, *it, mass_to_bucket(kData.ms[i]));
  }
  return m;
}

static inline double mixed_coord(uint64_t id, uint64_t multiplier) {
  constexpr uint64_t kMask = (uint64_t{1} << 52) - 1;
  constexpr uint64_t kBase = uint64_t{1} << 52;
  return static_cast<double>(kBase + ((id * multiplier) & kMask));
}

static inline Particle mixed_particle(uint64_t id) {
  double x = mixed_coord(id, 0x9e3779b97f4a7c15ULL);
  double y = -mixed_coord(id, 0xd1b54a32d192ed03ULL);
  double m = static_cast<double>((id * 7) % 10 + 1);
  return Particle{id, x, y, m};
}

static std::vector<uint64_t> mixed_active_ids() {
  std::vector<uint64_t> ids(kN);
  for (size_t i = 0; i < kN; ++i)
    ids[i] = kData.ids[i];
  return ids;
}

static inline void mixed_insert(ParticleMap &m, const Particle &p) {
  auto it = m.create_all(p.id, p.x, p.y, p.m);
  benchmark::DoNotOptimize(it);
}

static inline void mixed_insert(PartParticleMap &m, const Particle &p) {
  auto it = m.create(p.id, p.x, p.y, p.m);
  if (it != m.cend()) {
    m.insert<BySeq>(*it);
    m.insert<ByX>(*it);
    m.insert<ByY>(*it);
    part_insert_bucket(m, *it, mass_to_bucket(p.m));
  }
  benchmark::DoNotOptimize(it);
}

template <typename Map> static inline void mixed_insert_bmi(Map &m, const Particle &p) {
  auto result = m.insert(p);
  benchmark::DoNotOptimize(result.second);
}

static inline void mixed_remove(ParticleMap &m, uint64_t id) {
  auto it = m.find_primary(id);
  bool found = it != m.cend();
  if (found)
    m.remove(*it);
  benchmark::DoNotOptimize(found);
}

static inline void mixed_remove(PartParticleMap &m, uint64_t id) {
  auto it = m.find_primary(id);
  bool found = it != m.cend();
  if (found)
    m.remove(*it);
  benchmark::DoNotOptimize(found);
}

template <typename Map> static inline void mixed_remove_bmi(Map &m, uint64_t id) {
  auto &idx = m.template get<ById>();
  auto it = idx.find(id);
  bool found = it != idx.end();
  if (found)
    idx.erase(it);
  benchmark::DoNotOptimize(found);
}

// ---------------------------------------------------------------------------
// BM: Create (insert N elements)
// ---------------------------------------------------------------------------
static void BM_OptIndex_Create(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = std::make_unique<ParticleMap>();
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i)
      m->create_all(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_OptIndex_Create)->Unit(benchmark::kMicrosecond);

static void BM_BMI_Create(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = std::make_unique<ParticleBMI>();
    prepare_bmi(*m);
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i)
      m->insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_BMI_Create)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_Create(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = std::make_unique<ParticleBMIPool>();
    prepare_pool_bmi(*m);
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i)
      m->insert(Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PoolBMI_Create)->Unit(benchmark::kMicrosecond);

static void BM_PartOptIndex_Create(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = std::make_unique<PartParticleMap>();
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i) {
      auto it =
          m->create(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      m->insert<BySeq>(*it);
      m->insert<ByX>(*it);
      m->insert<ByY>(*it);
      part_insert_bucket(*m, *it, mass_to_bucket(kData.ms[i]));
    }
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PartOptIndex_Create)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Find by primary key
// ---------------------------------------------------------------------------
static void BM_OptIndex_FindPrimary(benchmark::State &state) {
  auto m = make_optindex();
  size_t i = 0;
  for (auto _ : state) {
    auto it = m->find_primary(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != m->cend()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_OptIndex_FindPrimary)->Unit(benchmark::kNanosecond);

static void BM_BMI_FindPrimary(benchmark::State &state) {
  auto m = make_bmi();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto it = idx.find(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != m->cend()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_BMI_FindPrimary)->Unit(benchmark::kNanosecond);

static void BM_PoolBMI_FindPrimary(benchmark::State &state) {
  auto m = make_bmi_pool();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto it = idx.find(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != m->cend()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_PoolBMI_FindPrimary)->Unit(benchmark::kNanosecond);

static void BM_PartOptIndex_FindPrimary(benchmark::State &state) {
  auto m = make_part_optindex();
  size_t i = 0;
  for (auto _ : state) {
    auto it = m->find_primary(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != m->cend()) {
      double x = it->x;
      benchmark::DoNotOptimize(x);
    }
    ++i;
  }
}
BENCHMARK(BM_PartOptIndex_FindPrimary)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM: Remove by primary key
// ---------------------------------------------------------------------------
static void BM_OptIndex_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_optindex();
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i) {
      auto it = m->find_primary(kData.ids[i]);
      if (it != m->cend())
        m->remove(*it);
    }
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_OptIndex_Remove)->Unit(benchmark::kMicrosecond);

static void BM_BMI_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_bmi();
    state.ResumeTiming();
    auto &idx = m->get<ById>();
    for (size_t i = 0; i < kN; ++i) {
      auto it = idx.find(kData.ids[i]);
      if (it != idx.end())
        idx.erase(it);
    }
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_BMI_Remove)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_bmi_pool();
    state.ResumeTiming();
    auto &idx = m->get<ById>();
    for (size_t i = 0; i < kN; ++i) {
      auto it = idx.find(kData.ids[i]);
      if (it != idx.end())
        idx.erase(it);
    }
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PoolBMI_Remove)->Unit(benchmark::kMicrosecond);

static void BM_PartOptIndex_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_part_optindex();
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i) {
      auto it = m->find_primary(kData.ids[i]);
      if (it != m->cend())
        m->remove(*it);
    }
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_PartOptIndex_Remove)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Bulk iteration (BySeq)
// ---------------------------------------------------------------------------
static void BM_OptIndex_BulkIterate(benchmark::State &state) {
  auto m = make_optindex();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<BySeq>())
      sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_OptIndex_BulkIterate)->Unit(benchmark::kMicrosecond);

static void BM_BMI_BulkIterate(benchmark::State &state) {
  auto m = make_bmi();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<BySeq>())
      sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_BMI_BulkIterate)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_BulkIterate(benchmark::State &state) {
  auto m = make_bmi_pool();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<BySeq>())
      sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PoolBMI_BulkIterate)->Unit(benchmark::kMicrosecond);

static void BM_PartOptIndex_BulkIterate(benchmark::State &state) {
  auto m = make_part_optindex();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<BySeq>())
      sum += p.x + p.y + p.m;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PartOptIndex_BulkIterate)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Ordered iteration (ByX)
// ---------------------------------------------------------------------------
static void BM_OptIndex_OrderedIterate(benchmark::State &state) {
  auto m = make_optindex();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_OptIndex_OrderedIterate)->Unit(benchmark::kMicrosecond);

static void BM_BMI_OrderedIterate(benchmark::State &state) {
  auto m = make_bmi();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_BMI_OrderedIterate)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_OrderedIterate(benchmark::State &state) {
  auto m = make_bmi_pool();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PoolBMI_OrderedIterate)->Unit(benchmark::kMicrosecond);

static void BM_PartOptIndex_OrderedIterate(benchmark::State &state) {
  auto m = make_part_optindex();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PartOptIndex_OrderedIterate)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Mass range (equal_range for ordered variants; bucket walk for partitioned)
// ---------------------------------------------------------------------------
static void BM_OptIndex_MassRange(benchmark::State &state) {
  auto m = make_optindex();
  size_t i = 0;
  for (auto _ : state) {
    double mass = kData.lookup_ms[i % kData.lookup_ms.size()];
    auto &idx = m->get<ByM>();
    auto [beg, end] = idx.equal_range(mass);
    size_t count = 0;
    for (auto it = beg; it != end; ++it)
      ++count;
    benchmark::DoNotOptimize(count);
    ++i;
  }
}
BENCHMARK(BM_OptIndex_MassRange)->Unit(benchmark::kNanosecond);

static void BM_BMI_MassRange(benchmark::State &state) {
  auto m = make_bmi();
  auto &idx = m->get<ByM>();
  size_t i = 0;
  for (auto _ : state) {
    double mass = kData.lookup_ms[i % kData.lookup_ms.size()];
    auto [beg, end] = idx.equal_range(mass);
    size_t count = 0;
    for (auto it = beg; it != end; ++it)
      ++count;
    benchmark::DoNotOptimize(count);
    ++i;
  }
}
BENCHMARK(BM_BMI_MassRange)->Unit(benchmark::kNanosecond);

static void BM_PoolBMI_MassRange(benchmark::State &state) {
  auto m = make_bmi_pool();
  auto &idx = m->get<ByM>();
  size_t i = 0;
  for (auto _ : state) {
    double mass = kData.lookup_ms[i % kData.lookup_ms.size()];
    auto [beg, end] = idx.equal_range(mass);
    size_t count = 0;
    for (auto it = beg; it != end; ++it)
      ++count;
    benchmark::DoNotOptimize(count);
    ++i;
  }
}
BENCHMARK(BM_PoolBMI_MassRange)->Unit(benchmark::kNanosecond);

static void BM_PartOptIndex_MassRange(benchmark::State &state) {
  auto m = make_part_optindex();
  size_t i = 0;
  for (auto _ : state) {
    int bkt = mass_to_bucket(kData.lookup_ms[i % kData.lookup_ms.size()]);
    size_t count = part_count_bucket(*m, bkt);
    benchmark::DoNotOptimize(count);
    ++i;
  }
}
BENCHMARK(BM_PartOptIndex_MassRange)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM: Modify mass — new mass always forced into a different bucket.
// Partitioned: modify<ReindexNone> writes the field, then unindex<Old> +
// insert<New>. Others: standard modify reindexing ByM.
// ---------------------------------------------------------------------------
static inline double next_bucket_mass(double old_mass, size_t i) {
  int old_bkt = mass_to_bucket(old_mass);
  int new_bkt = (old_bkt + 1 + (i % 9)) % 10;
  return static_cast<double>(new_bkt + 1);
}

static void BM_OptIndex_Modify(benchmark::State &state) {
  auto m = make_optindex();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = m->find_primary(id);
    if (it != m->cend()) {
      double nm = next_bucket_mass(it->m, i);
      m->modify<optindex::ReindexOnly<ByM>>(
          *it, [nm](Particle &p) { p.m = nm; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_OptIndex_Modify)->Unit(benchmark::kNanosecond);

static void BM_BMI_Modify(benchmark::State &state) {
  auto m = make_bmi();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = idx.find(id);
    if (it != idx.end()) {
      double nm = next_bucket_mass(it->m, i);
      idx.modify(it, [nm](Particle &p) { p.m = nm; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_BMI_Modify)->Unit(benchmark::kNanosecond);

static void BM_PoolBMI_Modify(benchmark::State &state) {
  auto m = make_bmi_pool();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = idx.find(id);
    if (it != idx.end()) {
      double nm = next_bucket_mass(it->m, i);
      idx.modify(it, [nm](Particle &p) { p.m = nm; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_PoolBMI_Modify)->Unit(benchmark::kNanosecond);

static void BM_PartOptIndex_Modify(benchmark::State &state) {
  auto m = make_part_optindex();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = m->find_primary(id);
    if (it != m->cend()) {
      int old_bkt = mass_to_bucket(it->m);
      int new_bkt = (old_bkt + 1 + (i % 9)) % 10;
      double nm = static_cast<double>(new_bkt + 1);
      m->modify<optindex::ReindexNone>(*it,
                                       [nm](Particle &p) { p.m = nm; });
      part_unindex_bucket(*m, *it, old_bkt);
      part_insert_bucket(*m, *it, new_bkt);
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_PartOptIndex_Modify)->Unit(benchmark::kNanosecond);

static inline void mixed_modify_mass(ParticleMap &m, uint64_t id, size_t i) {
  auto it = m.find_primary(id);
  if (it != m.cend()) {
    double nm = next_bucket_mass(it->m, i);
    bool ok = m.modify<optindex::ReindexOnly<ByM>>(
        *it, [nm](Particle &p) { p.m = nm; });
    benchmark::DoNotOptimize(ok);
  }
}

static inline void mixed_modify_mass(PartParticleMap &m, uint64_t id, size_t i) {
  auto it = m.find_primary(id);
  if (it != m.cend()) {
    int old_bkt = mass_to_bucket(it->m);
    int new_bkt = (old_bkt + 1 + (i % 9)) % 10;
    double nm = static_cast<double>(new_bkt + 1);
    bool ok = m.modify<optindex::ReindexNone>(
        *it, [nm](Particle &p) { p.m = nm; });
    part_unindex_bucket(m, *it, old_bkt);
    part_insert_bucket(m, *it, new_bkt);
    benchmark::DoNotOptimize(ok);
  }
}

template <typename Map>
static inline void mixed_modify_mass_bmi(Map &m, uint64_t id, size_t i) {
  auto &idx = m.template get<ById>();
  auto it = idx.find(id);
  if (it != idx.end()) {
    double nm = next_bucket_mass(it->m, i);
    bool ok = idx.modify(it, [nm](Particle &p) { p.m = nm; });
    benchmark::DoNotOptimize(ok);
  }
}

static inline double mixed_lookup(ParticleMap &m, uint64_t id) {
  auto it = m.find_primary(id);
  return it == m.cend() ? 0.0 : it->x;
}

static inline double mixed_lookup(PartParticleMap &m, uint64_t id) {
  auto it = m.find_primary(id);
  return it == m.cend() ? 0.0 : it->x;
}

template <typename Map> static inline double mixed_lookup_bmi(Map &m, uint64_t id) {
  auto &idx = m.template get<ById>();
  auto it = idx.find(id);
  return it == idx.end() ? 0.0 : it->x;
}

template <typename Map> static inline double mixed_scan_seq(Map &m) {
  double sum = 0.0;
  for (auto &p : m.template get<BySeq>())
    sum += p.x + p.y + p.m;
  return sum;
}

template <typename Map, typename Remove, typename Insert, typename Modify,
          typename Lookup>
static void run_mixed_steady_state(benchmark::State &state, Map &m,
                                   Remove remove_by_id,
                                   Insert insert_particle, Modify modify_mass,
                                   Lookup lookup_id) {
  auto active_ids = mixed_active_ids();
  uint64_t next_id = kN + 1;
  size_t cursor = 0;
  size_t op = 0;

  for (auto _ : state) {
    for (size_t round = 0; round < kMixedSteadyRounds; ++round) {
      
      for (size_t j = 0; j < kMixedSteadyBatch; ++j) {
        size_t slot = (cursor + j) % active_ids.size();
        remove_by_id(m, active_ids[slot]);

        Particle p = mixed_particle(next_id++);
        insert_particle(m, p);
        active_ids[slot] = p.id;
      }
      cursor = (cursor + kMixedSteadyBatch) % active_ids.size();
    }

    for (size_t j = 0; j < kMixedSteadyBatch; ++j) {
      size_t slot = (cursor + j) % active_ids.size();
      modify_mass(m, active_ids[slot], op++);
    }

    double lookup_sum = 0.0;
    for (size_t j = 0; j < kMixedSteadyBatch; ++j) {
      size_t slot = (cursor + kMixedSteadyBatch + j) % active_ids.size();
      lookup_sum += lookup_id(m, active_ids[slot]);
    }

    double scan_sum = mixed_scan_seq(m);
    benchmark::DoNotOptimize(lookup_sum);
    benchmark::DoNotOptimize(scan_sum);
    benchmark::DoNotOptimize(m.size());
  }
}

// ---------------------------------------------------------------------------
// BM: Mixed steady-state pressure. One measured operation keeps kN particles
// resident while doing batched primary removals, replacement inserts, mass
// updates, primary lookups, and one BySeq scan.
// ---------------------------------------------------------------------------
static void BM_OptIndex_MixedSteadyState(benchmark::State &state) {
  auto m = make_optindex();
  run_mixed_steady_state(
      state, *m, [](auto &map, uint64_t id) { mixed_remove(map, id); },
      [](auto &map, const Particle &p) { mixed_insert(map, p); },
      [](auto &map, uint64_t id, size_t i) { mixed_modify_mass(map, id, i); },
      [](auto &map, uint64_t id) { return mixed_lookup(map, id); });
}
BENCHMARK(BM_OptIndex_MixedSteadyState)->Unit(benchmark::kMicrosecond);

static void BM_BMI_MixedSteadyState(benchmark::State &state) {
  auto m = make_bmi();
  run_mixed_steady_state(
      state, *m,
      [](auto &map, uint64_t id) { mixed_remove_bmi(map, id); },
      [](auto &map, const Particle &p) { mixed_insert_bmi(map, p); },
      [](auto &map, uint64_t id, size_t i) {
        mixed_modify_mass_bmi(map, id, i);
      },
      [](auto &map, uint64_t id) { return mixed_lookup_bmi(map, id); });
}
BENCHMARK(BM_BMI_MixedSteadyState)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_MixedSteadyState(benchmark::State &state) {
  auto m = make_bmi_pool();
  run_mixed_steady_state(
      state, *m,
      [](auto &map, uint64_t id) { mixed_remove_bmi(map, id); },
      [](auto &map, const Particle &p) { mixed_insert_bmi(map, p); },
      [](auto &map, uint64_t id, size_t i) {
        mixed_modify_mass_bmi(map, id, i);
      },
      [](auto &map, uint64_t id) { return mixed_lookup_bmi(map, id); });
}
BENCHMARK(BM_PoolBMI_MixedSteadyState)->Unit(benchmark::kMicrosecond);

static void BM_PartOptIndex_MixedSteadyState(benchmark::State &state) {
  auto m = make_part_optindex();
  run_mixed_steady_state(
      state, *m, [](auto &map, uint64_t id) { mixed_remove(map, id); },
      [](auto &map, const Particle &p) { mixed_insert(map, p); },
      [](auto &map, uint64_t id, size_t i) { mixed_modify_mass(map, id, i); },
      [](auto &map, uint64_t id) { return mixed_lookup(map, id); });
}
BENCHMARK(BM_PartOptIndex_MixedSteadyState)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Modify X (reindex ByX)
// ---------------------------------------------------------------------------
static void BM_OptIndex_ModifyX(benchmark::State &state) {
  auto m = make_optindex();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = m->find_primary(id);
    if (it != m->cend()) {
      m->modify<optindex::ReindexOnly<ByX>>(
          *it, [](Particle &p) { p.x += 1.0; },
          [](Particle &p) { p.x -= 1.0; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_OptIndex_ModifyX)->Unit(benchmark::kNanosecond);

static void BM_BMI_ModifyX(benchmark::State &state) {
  auto m = make_bmi();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = idx.find(id);
    if (it != idx.end()) {
      idx.modify(
          it, [](Particle &p) { p.x += 1.0; },
          [](Particle &p) { p.x -= 1.0; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_BMI_ModifyX)->Unit(benchmark::kNanosecond);

static void BM_PoolBMI_ModifyX(benchmark::State &state) {
  auto m = make_bmi_pool();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = idx.find(id);
    if (it != idx.end()) {
      idx.modify(
          it, [](Particle &p) { p.x += 1.0; },
          [](Particle &p) { p.x -= 1.0; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_PoolBMI_ModifyX)->Unit(benchmark::kNanosecond);

static void BM_PartOptIndex_ModifyX(benchmark::State &state) {
  auto m = make_part_optindex();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = m->find_primary(id);
    if (it != m->cend()) {
      m->modify<optindex::ReindexOnly<ByX>>(
          *it, [](Particle &p) { p.x += 1.0; },
          [](Particle &p) { p.x -= 1.0; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_PartOptIndex_ModifyX)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM: Distinct mass-level walk
// ---------------------------------------------------------------------------
static void BM_OptIndex_LevelWalk(benchmark::State &state) {
  auto m = make_optindex();
  for (auto _ : state) {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    benchmark::DoNotOptimize(levels);
  }
}
BENCHMARK(BM_OptIndex_LevelWalk)->Unit(benchmark::kMicrosecond);

static void BM_BMI_LevelWalk(benchmark::State &state) {
  auto m = make_bmi();
  for (auto _ : state) {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    benchmark::DoNotOptimize(levels);
  }
}
BENCHMARK(BM_BMI_LevelWalk)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_LevelWalk(benchmark::State &state) {
  auto m = make_bmi_pool();
  for (auto _ : state) {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    benchmark::DoNotOptimize(levels);
  }
}
BENCHMARK(BM_PoolBMI_LevelWalk)->Unit(benchmark::kMicrosecond);

static void BM_PartOptIndex_LevelWalk(benchmark::State &state) {
  auto m = make_part_optindex();
  for (auto _ : state) {
    size_t levels = 0;
    if (!m->get<BktM1>().empty())  ++levels;
    if (!m->get<BktM2>().empty())  ++levels;
    if (!m->get<BktM3>().empty())  ++levels;
    if (!m->get<BktM4>().empty())  ++levels;
    if (!m->get<BktM5>().empty())  ++levels;
    if (!m->get<BktM6>().empty())  ++levels;
    if (!m->get<BktM7>().empty())  ++levels;
    if (!m->get<BktM8>().empty())  ++levels;
    if (!m->get<BktM9>().empty())  ++levels;
    if (!m->get<BktM10>().empty()) ++levels;
    benchmark::DoNotOptimize(levels);
  }
}
BENCHMARK(BM_PartOptIndex_LevelWalk)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
