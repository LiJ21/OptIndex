#include <benchmark/benchmark.h>

// ---------------------------------------------------------------------------
// MultiMap under test
// ---------------------------------------------------------------------------
#include "multimap.h"

// ---------------------------------------------------------------------------
// Boost.MultiIndex
// ---------------------------------------------------------------------------
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <functional>
#include <memory>
#include <random>
#include <vector>
namespace bmi = boost::multi_index;

// ---------------------------------------------------------------------------
// Domain object — intentionally large to favour row-wise access
// ---------------------------------------------------------------------------
struct Particle {
  uint64_t id;
  double x;
  double y;
  double m; // mass — non-unique
  // Padding to make object "large" — realistic for real entities
  double vx{}, vy{};
  double fx{}, fy{};
  double charge{};
  uint32_t flags{};
  uint32_t zone{};
  char name[32]{"a default name"};

  Particle(uint64_t id, double x, double y, double m)
      : id(id), x(x), y(y), m(m) {}
};

// ---------------------------------------------------------------------------
// MultiMap setup
// ---------------------------------------------------------------------------
struct IdHash {
  size_t operator()(uint64_t id) const { return std::hash<uint64_t>{}(id); }
};
struct IdEqual {
  bool operator()(uint64_t a, uint64_t b) const { return a == b; }
};

#ifdef SET_KN
static constexpr size_t kN = SET_KN;
#else
static constexpr size_t kN = 10'000;
#endif
static constexpr size_t kBuckets = kN;

struct ById {};
struct ByX {};
struct ByY {};
struct ByM {};
struct BySeq {};

#if defined(WITH_HASH) && WITH_HASH
using ParticlePrimaryIndex = fastmm::Unordered<fastmm::KeyFrom<&Particle::id>,
                                               IdHash, IdEqual, kBuckets>;
using BMIPriIndex =
    bmi::hashed_unique<bmi::tag<ById>,
                       bmi::member<Particle, uint64_t, &Particle::id>>;
#else
using ParticlePrimaryIndex =
    fastmm::Ordered<fastmm::KeyFrom<&Particle::id>, std::less<uint64_t>>;
using BMIPriIndex =
    bmi::ordered_unique<bmi::tag<ById>,
                        bmi::member<Particle, uint64_t, &Particle::id>>;
#endif

using ParticleMap = fastmm::FixedSizeMultiMap<
    Particle, kN, fastmm::Named<ParticlePrimaryIndex, ById>,
#if defined(REV_INDEX) && REV_INDEX
    fastmm::Named<fastmm::List, BySeq>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>, ByX>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>, ByY>,
    fastmm::Named<fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>,
                                           std::less<double>>,
                  ByM>
#else
    fastmm::Named<fastmm::List, BySeq>,
    fastmm::Named<fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>,
                                           std::less<double>>,
                  ByM>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>, ByY>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>, ByX>
#endif
    >;

// ---------------------------------------------------------------------------
// Boost.MultiIndex setup
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
// Shared test data — generated once, reused across benchmarks
// ---------------------------------------------------------------------------

struct TestData {
  std::vector<uint64_t> ids;
  std::vector<double> xs, ys, ms;
  std::vector<uint64_t> lookup_ids; // random subset for find benchmarks
  std::vector<double> lookup_ms;    // mass values for range benchmarks

  TestData() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> coord(-1000.0, 1000.0);
    // mass from small set — creates non-unique groups
    std::uniform_real_distribution<double> mass(1.0, 10.0);

    ids.resize(kN);
    xs.resize(kN);
    ys.resize(kN);
    ms.resize(kN);
    for (size_t i = 0; i < kN; ++i) {
      ids[i] = i + 1;
      xs[i] = coord(rng);
      ys[i] = coord(rng);
      ms[i] = std::round(mass(rng)); // integer masses — many repeats
    }

    // 256 random ids for lookup benchmarks
    lookup_ids.resize(256);
    std::uniform_int_distribution<size_t> pick(0, kN - 1);
    for (auto &id : lookup_ids)
      id = ids[pick(rng)];

    // 64 mass values for range benchmarks
    lookup_ms.resize(64);
    for (auto &m : lookup_ms)
      m = std::round(mass(rng));
  }
};

static const TestData kData;

// ---------------------------------------------------------------------------
// Helpers — populate containers
// ---------------------------------------------------------------------------
static auto make_multimap() {
  auto m = std::make_unique<ParticleMap>();
  for (size_t i = 0; i < kN; ++i)
    m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
  return m;
}

static auto make_multimap_primary_only() {
  auto m = std::make_unique<ParticleMap>();
  for (size_t i = 0; i < kN; ++i)
    m->insert<false>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
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

// ---------------------------------------------------------------------------
// BM: Create (insert N elements)
// ---------------------------------------------------------------------------
static void BM_MultiMap_Create(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = std::make_unique<ParticleMap>();
    state.ResumeTiming();
    for (size_t i = 0; i < kN; ++i)
      m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_MultiMap_Create)->Unit(benchmark::kMicrosecond);

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

// ---------------------------------------------------------------------------
// BM: Find by primary key (unique hash index)
// ---------------------------------------------------------------------------
static void BM_MultiMap_FindPrimary(benchmark::State &state) {
  auto m = make_multimap();
  size_t i = 0;
  for (auto _ : state) {
    auto it = m->find_primary(kData.lookup_ids[i % kData.lookup_ids.size()]);
    if (it != m->cend()) {
      double x = it->x;
      benchmark::DoNotOptimize(x); // force dereference — reads actual data
    }
    ++i;
  }
}
BENCHMARK(BM_MultiMap_FindPrimary)->Unit(benchmark::kNanosecond);

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

// ---------------------------------------------------------------------------
// BM: Remove by primary key
// ---------------------------------------------------------------------------
static void BM_MultiMap_Remove(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_multimap();
    state.ResumeTiming();

    for (size_t i = 0; i < kN; ++i) {
      auto it = m->find_primary(kData.ids[i]);
      if (it != m->cend())
        m->remove(m->to_mutable(*it));
    }
    benchmark::DoNotOptimize(m->size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}
BENCHMARK(BM_MultiMap_Remove)->Unit(benchmark::kMicrosecond);

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

// ---------------------------------------------------------------------------
// BM: Bulk iteration (list/sequenced index)
// ---------------------------------------------------------------------------
static void BM_MultiMap_BulkIterate(benchmark::State &state) {
  auto m = make_multimap();
  for (auto _ : state) {
    double sum = 0.0;
    auto &container = m->get<BySeq>(); // list index
    for (auto &p : container) {        // list index
      sum += p.x + p.y + p.m;
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_MultiMap_BulkIterate)->Unit(benchmark::kMicrosecond);

static void BM_BMI_BulkIterate(benchmark::State &state) {
  auto m = make_bmi();
  for (auto _ : state) {
    double sum = 0.0;
    auto &container = m->get<BySeq>(); // sequenced index
    for (auto &p : container) {
      sum += p.x + p.y + p.m;
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_BMI_BulkIterate)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_BulkIterate(benchmark::State &state) {
  auto m = make_bmi_pool();
  for (auto _ : state) {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container) {
      sum += p.x + p.y + p.m;
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PoolBMI_BulkIterate)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Non-unique range scan (mass level — price level analogue)
// ---------------------------------------------------------------------------
static void BM_MultiMap_MassRange(benchmark::State &state) {
  auto m = make_multimap();
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
BENCHMARK(BM_MultiMap_MassRange)->Unit(benchmark::kNanosecond);

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

// ---------------------------------------------------------------------------
// BM: Ordered iteration (x-index sorted traversal)
// ---------------------------------------------------------------------------
static void BM_MultiMap_OrderedIterate(benchmark::State &state) {
  auto m = make_multimap();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>()) { // x-ordered index
      sum += p.x;
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_MultiMap_OrderedIterate)->Unit(benchmark::kMicrosecond);

static void BM_BMI_OrderedIterate(benchmark::State &state) {
  auto m = make_bmi();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>()) {
      sum += p.x;
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_BMI_OrderedIterate)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_OrderedIterate(benchmark::State &state) {
  auto m = make_bmi_pool();
  for (auto _ : state) {
    double sum = 0.0;
    for (auto &p : m->get<ByX>()) {
      sum += p.x;
    }
    benchmark::DoNotOptimize(sum);
  }
}
BENCHMARK(BM_PoolBMI_OrderedIterate)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Modify (reindex x after mutation)
// ---------------------------------------------------------------------------
static void BM_MultiMap_Modify(benchmark::State &state) {
  auto m = make_multimap(); // via unique_ptr
  size_t i = 0;
  for (auto _ : state) {
    // Always modify mass (non-unique — never conflicts, no extra find needed)
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = m->find_primary(id);
    if (it != m->cend()) {
      m->modify<fastmm::ReindexOnlyByTag<ByM>>(
          *it, [i](Particle &p) { p.m = static_cast<double>(i % 10 + 1); });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_MultiMap_Modify)->Unit(benchmark::kNanosecond);

static void BM_BMI_Modify(benchmark::State &state) {
  auto m = make_bmi();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = idx.find(id);
    if (it != idx.end()) {
      // Force modification of all indices — fair comparison
      idx.modify(it,
                 [i](Particle &p) { p.m = static_cast<double>(i % 10 + 1); });
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
      idx.modify(it,
                 [i](Particle &p) { p.m = static_cast<double>(i % 10 + 1); });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_PoolBMI_Modify)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM: Modify (reindex x after mutation)
// ---------------------------------------------------------------------------
static void BM_MultiMap_ModifyX(benchmark::State &state) {
  auto m = make_multimap(); // via unique_ptr
  size_t i = 0;
  for (auto _ : state) {
    // Always modify mass (non-unique — never conflicts, no extra find needed)
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = m->find_primary(id);
    if (it != m->cend()) {
      m->modify<ByX>(
          *it, [](Particle &p) { p.x += 1.0; },
          [](Particle &p) { p.x -= 1.0; });
      // std::cout << it->x << std::endl;
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_MultiMap_ModifyX)->Unit(benchmark::kNanosecond);

static void BM_BMI_ModifyX(benchmark::State &state) {
  auto m = make_bmi();
  auto &idx = m->get<ById>();
  size_t i = 0;
  for (auto _ : state) {
    auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
    auto it = idx.find(id);
    if (it != idx.end()) {
      // Force modification of all indices — fair comparison
      idx.modify(
          it, [i](Particle &p) { p.x += 1.0; },
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
          it, [i](Particle &p) { p.x += 1.0; },
          [](Particle &p) { p.x -= 1.0; });
    }
    benchmark::DoNotOptimize(it);
    ++i;
  }
}
BENCHMARK(BM_PoolBMI_ModifyX)->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM: Distinct level walk (upper_bound pattern — price level sweep)
// ---------------------------------------------------------------------------
static void BM_MultiMap_LevelWalk(benchmark::State &state) {
  auto m = make_multimap();
  for (auto _ : state) {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    benchmark::DoNotOptimize(levels);
  }
}
BENCHMARK(BM_MultiMap_LevelWalk)->Unit(benchmark::kMicrosecond);

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

// ---------------------------------------------------------------------------
// BM: FastMM partial indexing — remove from one secondary index only
// ---------------------------------------------------------------------------
template <typename TTag>
static void BM_MultiMap_UnindexSecondary(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_multimap();
    state.ResumeTiming();

    size_t removed = 0;
    for (auto it = m->cbegin(); it != m->cend(); ++it)
      removed += m->unindex<TTag>(*it) ? 1 : 0;
    benchmark::DoNotOptimize(removed);
    benchmark::DoNotOptimize(m->size());

    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

static void BM_MultiMap_UnindexMissingByX(benchmark::State &state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto m = make_multimap_primary_only();
    state.ResumeTiming();

    size_t removed = 0;
    for (auto it = m->cbegin(); it != m->cend(); ++it)
      removed += m->unindex<ByX>(*it) ? 1 : 0;
    benchmark::DoNotOptimize(removed);
    benchmark::DoNotOptimize(m->size());

    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * kN);
}

BENCHMARK_TEMPLATE(BM_MultiMap_UnindexSecondary, ByX)
    ->Name("BM_MultiMap_UnindexByX")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_MultiMap_UnindexSecondary, ByM)
    ->Name("BM_MultiMap_UnindexByM")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_MultiMap_UnindexSecondary, BySeq)
    ->Name("BM_MultiMap_UnindexBySeq")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MultiMap_UnindexMissingByX)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM: Mixed workload — multi-phase churn
// Pattern per round:
//   1) create burst
//   2) find burst
//   3) modify burst
//   4) remove burst
//   5) create again
//   6) find again
//   7) modify again
//
// This simulates a container under repeated churn rather than a single pass.
// ---------------------------------------------------------------------------
static void BM_MultiMap_Mixed(benchmark::State &state) {
  constexpr size_t kInitial = kN / 2;
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  for (auto _ : state) {
    state.PauseTiming();

    auto m = std::make_unique<ParticleMap>(); // via unique_ptr to test move
                                              // semantics
    std::vector<uint64_t> live_ids;
    live_ids.reserve(kN);

    for (size_t i = 0; i < kInitial; ++i) {
      auto it =
          m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      if (it == m->cend())
        std::abort();
      live_ids.push_back(kData.ids[i]);
    }

    size_t next_create_idx = kInitial;
    size_t probe = 0;

    state.ResumeTiming();

    for (size_t round = 0; round < kRounds; ++round) {
      // 1) create burst
      for (size_t j = 0; j < kCreateBurst1 && next_create_idx < kN; ++j) {
        auto i = next_create_idx++;
        auto it = m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                  kData.ms[i]);
        if (it != m->cend())
          live_ids.push_back(kData.ids[i]);
      }

      // 2) find burst
      for (size_t j = 0; j < kFindBurst1 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = m->find_primary(id);
        benchmark::DoNotOptimize(it);
        if (it != m->cend()) {
          double x = it->x;
          benchmark::DoNotOptimize(x);
        }
      }
      probe += kFindBurst1;

      // 3) modify burst
      for (size_t j = 0; j < kModifyBurst1 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = m->find_primary(id);
        if (it != m->cend()) {
          m->modify<fastmm::ReindexOnly<3>>(*it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
          });
        }
      }
      probe += kModifyBurst1;

      // 4) remove burst
      // remove from the back for O(1) maintenance of live_ids
      for (size_t j = 0; j < kRemoveBurst && !live_ids.empty(); ++j) {
        uint64_t id = live_ids.back();
        live_ids.pop_back();
        auto it = m->find_primary(id);
        if (it != m->cend())
          m->remove(*it);
      }

      // 5) create again
      for (size_t j = 0; j < kCreateBurst2 && next_create_idx < kN; ++j) {
        auto i = next_create_idx++;
        auto it = m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                  kData.ms[i]);
        if (it != m->cend())
          live_ids.push_back(kData.ids[i]);
      }

      // 6) find again
      for (size_t j = 0; j < kFindBurst2 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = m->find_primary(id);
        benchmark::DoNotOptimize(it);
        if (it != m->cend()) {
          double x = it->x;
          benchmark::DoNotOptimize(x);
        }
      }
      probe += kFindBurst2;

      // 7) modify again
      for (size_t j = 0; j < kModifyBurst2 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = m->find_primary(id);
        if (it != m->cend()) {
          m->modify<fastmm::ReindexOnly<3>>(*it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
          });
        }
      }
      probe += kModifyBurst2;
    }

    benchmark::DoNotOptimize(m->size());
    benchmark::DoNotOptimize(live_ids.size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
}
BENCHMARK(BM_MultiMap_Mixed)->Unit(benchmark::kMicrosecond);

static void BM_BMI_Mixed(benchmark::State &state) {
  constexpr size_t kInitial = kN / 2;
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  for (auto _ : state) {
    state.PauseTiming();

    auto m = std::make_unique<ParticleBMI>();
    prepare_bmi(*m);
    std::vector<uint64_t> live_ids;
    live_ids.reserve(kN);

    for (size_t i = 0; i < kInitial; ++i) {
      auto [it, ok] = m->insert(
          Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
      if (!ok)
        std::abort();
      live_ids.push_back(kData.ids[i]);
    }

    size_t next_create_idx = kInitial;
    size_t probe = 0;

    state.ResumeTiming();

    auto &idx = m->get<ById>();

    for (size_t round = 0; round < kRounds; ++round) {
      // 1) create burst
      for (size_t j = 0; j < kCreateBurst1 && next_create_idx < kN; ++j) {
        auto i = next_create_idx++;
        auto [it, ok] =
            m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        if (ok)
          live_ids.push_back(kData.ids[i]);
      }

      // 2) find burst
      for (size_t j = 0; j < kFindBurst1 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        benchmark::DoNotOptimize(it);
        if (it != idx.end()) {
          double x = it->x;
          benchmark::DoNotOptimize(x);
        }
      }
      probe += kFindBurst1;

      // 3) modify burst
      for (size_t j = 0; j < kModifyBurst1 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
          });
        }
      }
      probe += kModifyBurst1;

      // 4) remove burst
      for (size_t j = 0; j < kRemoveBurst && !live_ids.empty(); ++j) {
        uint64_t id = live_ids.back();
        live_ids.pop_back();
        auto it = idx.find(id);
        if (it != idx.end())
          idx.erase(it);
      }

      // 5) create again
      for (size_t j = 0; j < kCreateBurst2 && next_create_idx < kN; ++j) {
        auto i = next_create_idx++;
        auto [it, ok] =
            m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        if (ok)
          live_ids.push_back(kData.ids[i]);
      }

      // 6) find again
      for (size_t j = 0; j < kFindBurst2 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        benchmark::DoNotOptimize(it);
        if (it != idx.end()) {
          double x = it->x;
          benchmark::DoNotOptimize(x);
        }
      }
      probe += kFindBurst2;

      // 7) modify again
      for (size_t j = 0; j < kModifyBurst2 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
          });
        }
      }
      probe += kModifyBurst2;
    }

    benchmark::DoNotOptimize(m->size());
    benchmark::DoNotOptimize(live_ids.size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
}
BENCHMARK(BM_BMI_Mixed)->Unit(benchmark::kMicrosecond);

static void BM_PoolBMI_Mixed(benchmark::State &state) {
  constexpr size_t kInitial = kN / 2;
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  for (auto _ : state) {
    state.PauseTiming();

    auto m = std::make_unique<ParticleBMIPool>();
    prepare_pool_bmi(*m);
    std::vector<uint64_t> live_ids;
    live_ids.reserve(kN);

    for (size_t i = 0; i < kInitial; ++i) {
      auto [it, ok] = m->insert(
          Particle{kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]});
      if (!ok)
        std::abort();
      live_ids.push_back(kData.ids[i]);
    }

    size_t next_create_idx = kInitial;
    size_t probe = 0;

    state.ResumeTiming();

    auto &idx = m->get<ById>();

    for (size_t round = 0; round < kRounds; ++round) {
      // 1) create burst
      for (size_t j = 0; j < kCreateBurst1 && next_create_idx < kN; ++j) {
        auto i = next_create_idx++;
        auto [it, ok] =
            m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        if (ok)
          live_ids.push_back(kData.ids[i]);
      }

      // 2) find burst
      for (size_t j = 0; j < kFindBurst1 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        benchmark::DoNotOptimize(it);
        if (it != idx.end()) {
          double x = it->x;
          benchmark::DoNotOptimize(x);
        }
      }
      probe += kFindBurst1;

      // 3) modify burst
      for (size_t j = 0; j < kModifyBurst1 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
          });
        }
      }
      probe += kModifyBurst1;

      // 4) remove burst
      for (size_t j = 0; j < kRemoveBurst && !live_ids.empty(); ++j) {
        uint64_t id = live_ids.back();
        live_ids.pop_back();
        auto it = idx.find(id);
        if (it != idx.end())
          idx.erase(it);
      }

      // 5) create again
      for (size_t j = 0; j < kCreateBurst2 && next_create_idx < kN; ++j) {
        auto i = next_create_idx++;
        auto [it, ok] =
            m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        if (ok)
          live_ids.push_back(kData.ids[i]);
      }

      // 6) find again
      for (size_t j = 0; j < kFindBurst2 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        benchmark::DoNotOptimize(it);
        if (it != idx.end()) {
          double x = it->x;
          benchmark::DoNotOptimize(x);
        }
      }
      probe += kFindBurst2;

      // 7) modify again
      for (size_t j = 0; j < kModifyBurst2 && !live_ids.empty(); ++j) {
        uint64_t id = live_ids[(probe + j) % live_ids.size()];
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
          });
        }
      }
      probe += kModifyBurst2;
    }

    benchmark::DoNotOptimize(m->size());
    benchmark::DoNotOptimize(live_ids.size());
    state.PauseTiming();
    m.reset();
    state.ResumeTiming();
  }
}
BENCHMARK(BM_PoolBMI_Mixed)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
BENCHMARK_MAIN();
