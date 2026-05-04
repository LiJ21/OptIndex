#include "extern/latency_bench.h"
#include "multimap.h"

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/tag.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/pool/pool_alloc.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ctime>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace bmi = boost::multi_index;

static void pin_to_core(int core) {
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    std::cerr << "warning: failed to pin to core " << core << ": "
              << std::strerror(errno) << "\n";
  }
#else
  (void)core;
  std::cerr << "warning: CPU pinning is only supported on Linux\n";
#endif
}

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

struct ById {};
struct ByX {};
struct ByY {};
struct ByM {};
struct BySeq {};

using ParticleMap = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Named<fastmm::List, BySeq>,
    fastmm::Named<fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>,
                                           std::less<double>>,
                  ByM>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::y>, std::less<double>>, ByY>,
    fastmm::Named<
        fastmm::Ordered<fastmm::KeyFrom<&Particle::x>, std::less<double>>,
        ByX>>;

using ParticleBMIAlloc =
    boost::fast_pool_allocator<Particle,
                               boost::default_user_allocator_new_delete,
                               boost::details::pool::null_mutex, kN>;

using BMIIndexedBy = bmi::indexed_by<
    bmi::hashed_unique<bmi::tag<ById>,
                       bmi::member<Particle, uint64_t, &Particle::id>>,
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

// ---------------------------------------------------------------------------
// Hot/Cold benchmark types
// Hot entities: primary hash + BySeq (active list) + ByM (spatial bucket).
// Cold entities: primary hash only (FastMM partial-index; BMI: cold_map).
// ---------------------------------------------------------------------------
using HotBMIIndexedBy = bmi::indexed_by<
    bmi::hashed_unique<bmi::tag<ById>,
                       bmi::member<Particle, uint64_t, &Particle::id>>,
    bmi::sequenced<bmi::tag<BySeq>>,
    bmi::ordered_non_unique<bmi::tag<ByM>,
                            bmi::member<Particle, double, &Particle::m>>>;
using HotOnlyBMI = bmi::multi_index_container<Particle, HotBMIIndexedBy>;
using HotOnlyBMIPoolAlloc =
    boost::fast_pool_allocator<Particle,
                               boost::default_user_allocator_new_delete,
                               boost::details::pool::null_mutex, kN>;
using HotOnlyBMIPool =
    bmi::multi_index_container<Particle, HotBMIIndexedBy, HotOnlyBMIPoolAlloc>;

// 3-index FastMM map matching the slot footprint of HotOnlyBMI nodes.
// Omits ByX and ByY so slot_size drops from 240 to 176 bytes.
using HotColdParticleMap = fastmm::FixedSizeMultiMap<
    Particle, kN,
    fastmm::Unordered<fastmm::KeyFrom<&Particle::id>, IdHash, IdEqual,
                      kBuckets>,
    fastmm::Named<fastmm::List, BySeq>,
    fastmm::Named<fastmm::OrderedNonUnique<fastmm::KeyFrom<&Particle::m>,
                                           std::less<double>>,
                  ByM>>;

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
    for (auto &m : lookup_ms)
      m = std::round(mass(rng));
  }
};

static const TestData kData;

static auto make_multimap() {
  auto m = std::make_shared<ParticleMap>();
  for (size_t i = 0; i < kN; ++i)
    m->insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
  return m;
}

static auto make_bmi() {
  auto m = std::make_shared<ParticleBMI>();
  m->reserve(kN);
  for (size_t i = 0; i < kN; ++i)
    m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
  return m;
}

static auto make_bmi_pool() {
  auto m = std::make_shared<ParticleBMIPool>();
  m->reserve(kN);
  m->get<ById>().rehash(kN);
  for (size_t i = 0; i < kN; ++i)
    m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
  return m;
}

static std::string format_ns(double ns) {
  char buf[32];
  double v = std::abs(ns);
  if (v < 1000.0)
    std::snprintf(buf, sizeof(buf), "%.1f ns", ns);
  else if (v < 1'000'000.0)
    std::snprintf(buf, sizeof(buf), "%.1f us", ns / 1000.0);
  else if (v < 1'000'000'000.0)
    std::snprintf(buf, sizeof(buf), "%.1f ms", ns / 1'000'000.0);
  else
    std::snprintf(buf, sizeof(buf), "%.2f s", ns / 1'000'000'000.0);
  return buf;
}

static std::string format_ns(int64_t v) {
  return format_ns(static_cast<double>(v));
}

static std::string format_throughput(double mean_ns) {
  if (mean_ns <= 0.0)
    return "n/a";
  double ops_per_s = 1e9 / mean_ns;
  char buf[32];
  if (ops_per_s >= 1e9)
    std::snprintf(buf, sizeof(buf), "%.2f Gops/s", ops_per_s / 1e9);
  else if (ops_per_s >= 1e6)
    std::snprintf(buf, sizeof(buf), "%.1f Mops/s", ops_per_s / 1e6);
  else if (ops_per_s >= 1e3)
    std::snprintf(buf, sizeof(buf), "%.1f Kops/s", ops_per_s / 1e3);
  else
    std::snprintf(buf, sizeof(buf), "%.0f ops/s", ops_per_s);
  return buf;
}

struct PercentileReport {
  std::string name;
  int64_t count{};
  double mean{};
  double sample_stddev{};
  double mean_stderr{};
  int64_t p50{}, p90{}, p95{}, p99{}, p999{}, max{};

  static PercentileReport from(const std::string &name, hdr_histogram *h) {
    return {name,
            h->total_count,
            hdr_mean(h),
            hdr_stddev(h),
            h->total_count > 0 ? hdr_stddev(h) / std::sqrt(h->total_count)
                               : 0.0,
            hdr_value_at_percentile(h, 50.0),
            hdr_value_at_percentile(h, 90.0),
            hdr_value_at_percentile(h, 95.0),
            hdr_value_at_percentile(h, 99.0),
            hdr_value_at_percentile(h, 99.9),
            hdr_max(h)};
  }

  void print_text(std::ostream &os) const {
    os << std::left << std::setw(28) << name << " n=" << std::setw(8) << count
       << " mean=" << std::fixed << std::setprecision(1) << std::setw(9) << mean
       << " sample_stddev=" << std::setw(9) << sample_stddev
       << " mean_stderr=" << std::setw(9) << mean_stderr
       << " p50=" << std::setw(8) << p50 << " p90=" << std::setw(8) << p90
       << " p95=" << std::setw(8) << p95 << " p99=" << std::setw(8) << p99
       << " p99.9=" << std::setw(8) << p999 << " max=" << std::setw(8) << max
       << " (ns)\n";
  }

  void print_json(std::ostream &os) const {
    os << "{\"name\":\"" << name << "\",\"count\":" << count
       << ",\"mean\":" << std::fixed << std::setprecision(1) << mean
       << ",\"sample_stddev\":" << sample_stddev
       << ",\"mean_stderr\":" << mean_stderr << ",\"p50\":" << p50
       << ",\"p90\":" << p90 << ",\"p95\":" << p95 << ",\"p99\":" << p99
       << ",\"p999\":" << p999 << ",\"max\":" << max << "}";
  }

  void print_gbench(std::ostream &os, int name_w, int time_w, int thput_w,
                    int iters_w) const {
    os << std::left << std::setw(name_w) << name << std::right
       << std::setw(time_w) << format_ns(mean) << std::setw(time_w)
       << format_ns(p50) << std::setw(time_w) << format_ns(p90)
       << std::setw(time_w) << format_ns(p99) << std::setw(time_w)
       << format_ns(p999) << std::setw(time_w) << format_ns(max)
       << std::setw(thput_w) << format_throughput(mean)
       << std::setw(iters_w) << count << "\n";
  }
};

struct HistogramBucket {
  int64_t value{};
  int64_t lowest_equivalent_value{};
  int64_t highest_equivalent_value{};
  int64_t count{};
  int64_t cumulative_count{};
};

struct DetailedReport {
  PercentileReport summary;
  std::vector<HistogramBucket> histogram;

  static DetailedReport from(const std::string &name, hdr_histogram *h) {
    DetailedReport report{PercentileReport::from(name, h), {}};

    hdr_iter iter;
    hdr_iter_recorded_init(&iter, h);
    while (hdr_iter_next(&iter)) {
      report.histogram.push_back({iter.value, iter.lowest_equivalent_value,
                                  iter.highest_equivalent_value, iter.count,
                                  iter.cumulative_count});
    }

    return report;
  }
};

static void write_json_string(std::ostream &os, const std::string &value) {
  os << '"';
  for (char c : value) {
    switch (c) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << '"';
}

static void write_summary_json(std::ostream &os, const PercentileReport &r) {
  os << "{\"name\":";
  write_json_string(os, r.name);
  os << ",\"count\":" << r.count << ",\"mean\":" << std::fixed
     << std::setprecision(1) << r.mean
     << ",\"sample_stddev\":" << r.sample_stddev
     << ",\"mean_stderr\":" << r.mean_stderr << ",\"p50\":" << r.p50
     << ",\"p90\":" << r.p90 << ",\"p95\":" << r.p95 << ",\"p99\":" << r.p99
     << ",\"p999\":" << r.p999 << ",\"max\":" << r.max << "}";
}

static void
write_histogram_json_file(const std::string &path,
                          const std::vector<DetailedReport> &reports) {
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("failed to open histogram json output: " + path);

  out << "{\n  \"unit\":\"ns\",\n  \"benchmarks\":[";
  for (size_t i = 0; i < reports.size(); ++i) {
    const auto &report = reports[i];
    out << (i ? "," : "") << "\n    {\n      \"summary\":";
    write_summary_json(out, report.summary);
    out << ",\n      \"histogram\":[";
    for (size_t j = 0; j < report.histogram.size(); ++j) {
      const auto &bucket = report.histogram[j];
      out << (j ? "," : "") << "\n        {\"value\":" << bucket.value
          << ",\"lowest_equivalent_value\":" << bucket.lowest_equivalent_value
          << ",\"highest_equivalent_value\":" << bucket.highest_equivalent_value
          << ",\"count\":" << bucket.count
          << ",\"cumulative_count\":" << bucket.cumulative_count << "}";
    }
    out << "\n      ]\n    }";
  }
  out << "\n  ]\n}\n";
}

struct RegisteredLatencyBench {
  std::string name;
  std::vector<std::string> output_names;
  std::function<std::vector<DetailedReport>()> run;
};

static std::vector<RegisteredLatencyBench> &registry() {
  static std::vector<RegisteredLatencyBench> benches;
  return benches;
}

static std::string g_filter;
static int64_t g_measure_iters_override = 0;

static bool should_register(const char *name) {
  const std::string bench_name{name};
  return g_filter.empty() || bench_name.find(g_filter) != std::string::npos ||
         g_filter.find(bench_name) != std::string::npos;
}

static bool should_run(const RegisteredLatencyBench &bench,
                       const std::string &filter) {
  if (filter.empty() || bench.name.find(filter) != std::string::npos)
    return true;

  for (const auto &name : bench.output_names) {
    if (name.find(filter) != std::string::npos)
      return true;
  }
  return false;
}

static bool should_keep_report(const DetailedReport &report,
                               const std::string &filter) {
  return filter.empty() ||
         report.summary.name.find(filter) != std::string::npos;
}

template <size_t N>
static bool should_register_any(const char *const (&names)[N]) {
  for (const char *name : names) {
    if (should_register(name))
      return true;
  }
  return false;
}

static void apply_iteration_override(lb::BenchConfig &cfg) {
  if (g_measure_iters_override > 0)
    cfg.measure_iters = g_measure_iters_override;
}

template <typename Fn>
static void register_latency_bench(const char *name, lb::BenchConfig cfg,
                                   Fn &&fn) {
  std::string bench_name{name};
  registry().push_back(
      {bench_name,
       {bench_name},
       [bench_name, cfg, fn = std::forward<Fn>(fn)]() mutable {
         lb::LatencyBench<> bench(cfg);
         bench.run(fn);
         return std::vector<DetailedReport>{
             DetailedReport::from(bench_name, bench.histogram())};
       }});
}

template <typename Setup, typename Fn>
static void register_latency_bench(const char *name, lb::BenchConfig cfg,
                                   Setup &&setup, Fn &&fn) {
  std::string bench_name{name};
  registry().push_back(
      {bench_name,
       {bench_name},
       [bench_name, cfg, setup = std::forward<Setup>(setup),
        fn = std::forward<Fn>(fn)]() mutable {
         lb::LatencyBench<> bench(cfg);
         bench.run(setup, fn);
         return std::vector<DetailedReport>{
             DetailedReport::from(bench_name, bench.histogram())};
       }});
}

static lb::BenchConfig single_op_cfg() {
  lb::BenchConfig cfg;
  cfg.warmup_iters = 10'000;
  cfg.measure_iters = 100'000;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig container_op_cfg() {
  lb::BenchConfig cfg;
  cfg.warmup_iters = 2;
  cfg.measure_iters = 50;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig extended_mixed_cfg() {
  lb::BenchConfig cfg;
  cfg.warmup_iters = 1;
  cfg.measure_iters = 5;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig pass_cfg() {
  lb::BenchConfig cfg;
  cfg.warmup_iters = 20;
  cfg.measure_iters = 300;
  apply_iteration_override(cfg);
  return cfg;
}

static lb::BenchConfig level_walk_cfg() {
  lb::BenchConfig cfg;
  cfg.warmup_iters = 100;
  cfg.measure_iters = 1'000;
  apply_iteration_override(cfg);
  return cfg;
}

static void register_create_benches() {
  auto cfg = container_op_cfg();
  register_latency_bench(
      "MultiMap_Create", cfg, [] { return std::make_unique<ParticleMap>(); },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
          m->template insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                   kData.ms[i]);
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "DefaultBMI_Create", cfg,
      [] {
        auto m = std::make_unique<ParticleBMI>();
        m->reserve(kN);
        return m;
      },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
          m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "PoolBMI_Create", cfg,
      [] {
        auto m = std::make_unique<ParticleBMIPool>();
        m->reserve(kN);
        m->template get<ById>().rehash(kN);
        return m;
      },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i)
          m->emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        lb::do_not_optimize(m->size());
      });
}

static void register_find_primary_benches() {
  auto cfg = single_op_cfg();

  register_latency_bench("MultiMap_FindPrimary", cfg,
                         [m = make_multimap(), i = size_t{0}]() mutable {
                           auto it = m->find_primary(
                               kData.lookup_ids[i % kData.lookup_ids.size()]);
                           if (it != m->cend())
                             lb::do_not_optimize(it->x);
                           ++i;
                         });

  register_latency_bench(
      "DefaultBMI_FindPrimary", cfg, [m = make_bmi(), i = size_t{0}]() mutable {
        auto &idx = m->get<ById>();
        auto it = idx.find(kData.lookup_ids[i % kData.lookup_ids.size()]);
        if (it != idx.end())
          lb::do_not_optimize(it->x);
        ++i;
      });

  register_latency_bench("PoolBMI_FindPrimary", cfg,
                         [m = make_bmi_pool(), i = size_t{0}]() mutable {
                           auto &idx = m->get<ById>();
                           auto it = idx.find(
                               kData.lookup_ids[i % kData.lookup_ids.size()]);
                           if (it != idx.end())
                             lb::do_not_optimize(it->x);
                           ++i;
                         });
}

static void register_remove_benches() {
  auto cfg = container_op_cfg();

  register_latency_bench(
      "MultiMap_Remove", cfg, [] { return make_multimap(); },
      [](auto &m) {
        for (size_t i = 0; i < kN; ++i) {
          m->remove(kData.ids[i]);
        }
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "DefaultBMI_Remove", cfg, [] { return make_bmi(); },
      [](auto &m) {
        auto &idx = m->template get<ById>();
        for (size_t i = 0; i < kN; ++i) {
          auto it = idx.find(kData.ids[i]);
          if (it != idx.end())
            idx.erase(it);
        }
        lb::do_not_optimize(m->size());
      });

  register_latency_bench(
      "PoolBMI_Remove", cfg, [] { return make_bmi_pool(); },
      [](auto &m) {
        auto &idx = m->template get<ById>();
        for (size_t i = 0; i < kN; ++i) {
          auto it = idx.find(kData.ids[i]);
          if (it != idx.end())
            idx.erase(it);
        }
        lb::do_not_optimize(m->size());
      });
}

static void register_bulk_iterate_benches() {
  auto cfg = pass_cfg();

  register_latency_bench("MultiMap_BulkIterate", cfg, [m = make_multimap()] {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container)
      sum += p.x + p.y + p.m;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("DefaultBMI_BulkIterate", cfg, [m = make_bmi()] {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container)
      sum += p.x + p.y + p.m;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("PoolBMI_BulkIterate", cfg, [m = make_bmi_pool()] {
    double sum = 0.0;
    auto &container = m->get<BySeq>();
    for (auto &p : container)
      sum += p.x + p.y + p.m;
    lb::do_not_optimize(sum);
  });
}

static void register_mass_range_benches() {
  auto cfg = single_op_cfg();

  register_latency_bench("MultiMap_MassRange", cfg,
                         [m = make_multimap(), i = size_t{0}]() mutable {
                           double mass =
                               kData.lookup_ms[i % kData.lookup_ms.size()];
                           auto &idx = m->get<2>();
                           auto [beg, end] = idx.equal_range(mass);
                           size_t count = 0;
                           for (auto it = beg; it != end; ++it)
                             ++count;
                           lb::do_not_optimize(count);
                           ++i;
                         });

  register_latency_bench(
      "DefaultBMI_MassRange", cfg, [m = make_bmi(), i = size_t{0}]() mutable {
        double mass = kData.lookup_ms[i % kData.lookup_ms.size()];
        auto &idx = m->get<ByM>();
        auto [beg, end] = idx.equal_range(mass);
        size_t count = 0;
        for (auto it = beg; it != end; ++it)
          ++count;
        lb::do_not_optimize(count);
        ++i;
      });

  register_latency_bench(
      "PoolBMI_MassRange", cfg, [m = make_bmi_pool(), i = size_t{0}]() mutable {
        double mass = kData.lookup_ms[i % kData.lookup_ms.size()];
        auto &idx = m->get<ByM>();
        auto [beg, end] = idx.equal_range(mass);
        size_t count = 0;
        for (auto it = beg; it != end; ++it)
          ++count;
        lb::do_not_optimize(count);
        ++i;
      });
}

static void register_ordered_iterate_benches() {
  auto cfg = pass_cfg();

  register_latency_bench("MultiMap_OrderedIterate", cfg, [m = make_multimap()] {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("DefaultBMI_OrderedIterate", cfg, [m = make_bmi()] {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });

  register_latency_bench("PoolBMI_OrderedIterate", cfg, [m = make_bmi_pool()] {
    double sum = 0.0;
    for (auto &p : m->get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });
}

static void register_modify_benches() {
  auto cfg = single_op_cfg();

  register_latency_bench(
      "MultiMap_Modify", cfg, [m = make_multimap(), i = size_t{0}]() mutable {
        auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
        auto it = m->find_primary(id);
        if (it != m->cend()) {
          m->modify<fastmm::ReindexOnlyByTag<ByM>>(
              *it, [i](Particle &p) { p.m = static_cast<double>(i % 10 + 1); });
        }
        lb::do_not_optimize(it);
        ++i;
      });

  register_latency_bench(
      "DefaultBMI_Modify", cfg, [m = make_bmi(), i = size_t{0}]() mutable {
        auto &idx = m->get<ById>();
        auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(
              it, [i](Particle &p) { p.m = static_cast<double>(i % 10 + 1); });
        }
        lb::do_not_optimize(it);
        ++i;
      });

  register_latency_bench(
      "PoolBMI_Modify", cfg, [m = make_bmi_pool(), i = size_t{0}]() mutable {
        auto &idx = m->get<ById>();
        auto id = kData.lookup_ids[i % kData.lookup_ids.size()];
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(
              it, [i](Particle &p) { p.m = static_cast<double>(i % 10 + 1); });
        }
        lb::do_not_optimize(it);
        ++i;
      });
}

static void register_level_walk_benches() {
  auto cfg = level_walk_cfg();

  register_latency_bench("MultiMap_LevelWalk", cfg, [m = make_multimap()] {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    lb::do_not_optimize(levels);
  });

  register_latency_bench("DefaultBMI_LevelWalk", cfg, [m = make_bmi()] {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    lb::do_not_optimize(levels);
  });

  register_latency_bench("PoolBMI_LevelWalk", cfg, [m = make_bmi_pool()] {
    auto &idx = m->get<ByM>();
    size_t levels = 0;
    for (auto it = idx.begin(); it != idx.end(); it = idx.upper_bound(it->m))
      ++levels;
    lb::do_not_optimize(levels);
  });
}

template <typename Map> struct MixedState {
  std::unique_ptr<Map> m;
  std::vector<uint64_t> live_ids;
  size_t next_create_idx{};
  size_t probe{};
};

struct MixedOpHistograms {
  hdr_histogram *insert = nullptr;
  hdr_histogram *find_primary = nullptr;
  hdr_histogram *mass_range = nullptr;
  hdr_histogram *modify = nullptr;
  hdr_histogram *remove = nullptr;
  hdr_histogram *ordered_iterate = nullptr;

  explicit MixedOpHistograms(const lb::BenchConfig &cfg) {
    init(cfg, insert);
    init(cfg, find_primary);
    init(cfg, mass_range);
    init(cfg, modify);
    init(cfg, remove);
    init(cfg, ordered_iterate);
  }

  ~MixedOpHistograms() {
    hdr_close(insert);
    hdr_close(find_primary);
    hdr_close(mass_range);
    hdr_close(modify);
    hdr_close(remove);
    hdr_close(ordered_iterate);
  }

  MixedOpHistograms(const MixedOpHistograms &) = delete;
  MixedOpHistograms &operator=(const MixedOpHistograms &) = delete;

  void reset() {
    hdr_reset(insert);
    hdr_reset(find_primary);
    hdr_reset(mass_range);
    hdr_reset(modify);
    hdr_reset(remove);
    hdr_reset(ordered_iterate);
  }

private:
  static void init(const lb::BenchConfig &cfg, hdr_histogram *&hist) {
    if (hdr_init(cfg.hist_min, cfg.hist_max, cfg.sig_digits, &hist))
      throw std::runtime_error("hdr_init failed");
  }
};

struct HotColdOpHistograms {
  hdr_histogram *transition_in{};   // cold→hot (warm up)
  hdr_histogram *transition_out{};  // hot→cold (cool down)
  hdr_histogram *update_hot{};      // modify hot entity (reindex ByM)
  hdr_histogram *query_bucket{};    // equal_range on ByM hot index
  hdr_histogram *churn_insert{};    // insert new cold entity
  hdr_histogram *churn_remove{};    // remove cold entity

  explicit HotColdOpHistograms(const lb::BenchConfig &cfg) {
    init(cfg, transition_in);
    init(cfg, transition_out);
    init(cfg, update_hot);
    init(cfg, query_bucket);
    init(cfg, churn_insert);
    init(cfg, churn_remove);
  }

  ~HotColdOpHistograms() {
    hdr_close(transition_in);
    hdr_close(transition_out);
    hdr_close(update_hot);
    hdr_close(query_bucket);
    hdr_close(churn_insert);
    hdr_close(churn_remove);
  }

  HotColdOpHistograms(const HotColdOpHistograms &) = delete;
  HotColdOpHistograms &operator=(const HotColdOpHistograms &) = delete;

  void reset() {
    hdr_reset(transition_in);
    hdr_reset(transition_out);
    hdr_reset(update_hot);
    hdr_reset(query_bucket);
    hdr_reset(churn_insert);
    hdr_reset(churn_remove);
  }

private:
  static void init(const lb::BenchConfig &cfg, hdr_histogram *&h) {
    if (hdr_init(cfg.hist_min, cfg.hist_max, cfg.sig_digits, &h))
      throw std::runtime_error("hdr_init failed");
  }
};

template <typename Fn>
static void record_mixed_op(hdr_histogram *hist, Fn &&fn) {
  if (!hist) {
    fn();
    return;
  }

  lb::clobber_memory();
  auto t0 = lb::ChronoClock::now();
  fn();
  auto t1 = lb::ChronoClock::now();
  lb::clobber_memory();
  hdr_record_value(hist, lb::ChronoClock::delta_ns(t0, t1));
}

template <typename Setup, typename Fn>
static void register_mixed_op_latency_bench(const char *base_name,
                                            lb::BenchConfig cfg, Setup &&setup,
                                            Fn &&fn) {
  const std::string base{base_name};
  std::vector<std::string> output_names{
      base + "_Insert", base + "_FindPrimary", base + "_Modify",
      base + "_Remove", base + "_MassRange",   base + "_OrderedIterate"};

  registry().push_back(
      {base + "_Ops", output_names,
       [output_names, cfg, setup = std::forward<Setup>(setup),
        fn = std::forward<Fn>(fn)]() mutable {
         MixedOpHistograms histograms(cfg);

         for (int64_t i = 0; i < cfg.warmup_iters; ++i) {
           auto state = setup();
           lb::clobber_memory();
           fn(state, nullptr);
           lb::clobber_memory();
         }

         histograms.reset();
         for (int64_t i = 0; i < cfg.measure_iters; ++i) {
           auto state = setup();
           lb::clobber_memory();
           fn(state, &histograms);
           lb::clobber_memory();
         }

         std::vector<DetailedReport> reports;
         auto add_if_recorded = [&reports](const std::string &name,
                                           hdr_histogram *hist) {
           if (hist->total_count > 0)
             reports.push_back(DetailedReport::from(name, hist));
         };
         add_if_recorded(output_names[0], histograms.insert);
         add_if_recorded(output_names[1], histograms.find_primary);
         add_if_recorded(output_names[2], histograms.modify);
         add_if_recorded(output_names[3], histograms.remove);
         add_if_recorded(output_names[4], histograms.mass_range);
         add_if_recorded(output_names[5], histograms.ordered_iterate);
         return reports;
       }});
}

template <typename Map, typename MakeMap, typename InsertInitial>
static auto make_mixed_state(MakeMap make_map, InsertInitial insert_initial) {
  constexpr size_t kInitial = kN / 2;
  auto state = MixedState<Map>{make_map(), {}, kInitial, 0};
  state.live_ids.reserve(kN);
  for (size_t i = 0; i < kInitial; ++i) {
    insert_initial(*state.m, i);
    state.live_ids.push_back(i + 1);
  }
  return state;
}

template <typename Map, typename MakeMap, typename InsertInitial>
static auto make_mixed_state(size_t initial, MakeMap make_map,
                             InsertInitial insert_initial) {
  if (initial > kN)
    initial = kN;

  auto state = MixedState<Map>{make_map(), {}, initial, 0};
  state.live_ids.reserve(kN);
  for (size_t i = 0; i < initial; ++i) {
    insert_initial(*state.m, i);
    state.live_ids.push_back(i + 1);
  }
  return state;
}

static double generated_x(size_t i) {
  return i < kN ? kData.xs[i]
                : static_cast<double>((i * 48271) % 200000) / 100.0 - 1000.0;
}

static double generated_y(size_t i) {
  return i < kN ? kData.ys[i]
                : static_cast<double>((i * 69621) % 200000) / 100.0 - 1000.0;
}

static double generated_m(size_t i) {
  return i < kN ? kData.ms[i] : static_cast<double>((i % 10) + 1);
}

static double mixed_mass_probe(size_t i) {
  return static_cast<double>((i % 10) + 1);
}

static void multimap_insert_mixed(MixedState<ParticleMap> &state,
                                  MixedOpHistograms *histograms) {
  auto &m = *state.m;
  auto i = state.next_create_idx++;
  auto it = m.cend();
  record_mixed_op(histograms ? histograms->insert : nullptr, [&] {
    it = m.insert<true>(i + 1, generated_x(i), generated_y(i), generated_m(i));
  });
  if (it != m.cend())
    state.live_ids.push_back(i + 1);
}

static void multimap_find_mixed(MixedState<ParticleMap> &state, size_t offset,
                                MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &m = *state.m;
  uint64_t id = state.live_ids[(state.probe + offset) % state.live_ids.size()];
  record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
    auto it = m.find_primary(id);
    lb::do_not_optimize(it);
    if (it != m.cend())
      lb::do_not_optimize(it->x);
  });
}

static void multimap_find_hot_mixed(MixedState<ParticleMap> &state,
                                    size_t offset,
                                    MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &m = *state.m;
  size_t hot = std::max<size_t>(1, state.live_ids.size() / 10);
  size_t idx = (offset % 10 == 0)
                   ? (state.probe + offset) % state.live_ids.size()
                   : (state.probe + offset) % hot;
  uint64_t id = state.live_ids[idx];
  record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
    auto it = m.find_primary(id);
    lb::do_not_optimize(it);
    if (it != m.cend())
      lb::do_not_optimize(it->x);
  });
}

static void multimap_modify_mixed(MixedState<ParticleMap> &state, size_t offset,
                                  size_t salt, MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &m = *state.m;
  uint64_t id = state.live_ids[(state.probe + offset) % state.live_ids.size()];
  record_mixed_op(histograms ? histograms->modify : nullptr, [&] {
    auto it = m.find_primary(id);
    if (it != m.cend()) {
      m.modify<fastmm::ReindexOnly<2>>(
          *it, [salt](Particle &p) { p.m = mixed_mass_probe(salt); });
    }
  });
}

static void multimap_remove_back_mixed(MixedState<ParticleMap> &state,
                                       MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &m = *state.m;
  uint64_t id = state.live_ids.back();
  state.live_ids.pop_back();
  record_mixed_op(histograms ? histograms->remove : nullptr, [&] {
    auto it = m.find_primary(id);
    if (it != m.cend())
      m.remove(*it);
  });
}

static void multimap_mass_range_mixed(MixedState<ParticleMap> &state,
                                      size_t offset,
                                      MixedOpHistograms *histograms) {
  auto &m = *state.m;
  double mass = mixed_mass_probe(state.probe + offset);
  record_mixed_op(histograms ? histograms->mass_range : nullptr, [&] {
    auto &idx = m.get<ByM>();
    auto [beg, end] = idx.equal_range(mass);
    size_t count = 0;
    for (auto it = beg; it != end; ++it)
      ++count;
    lb::do_not_optimize(count);
  });
}

static void multimap_ordered_iterate_mixed(MixedState<ParticleMap> &state,
                                           MixedOpHistograms *histograms) {
  auto &m = *state.m;
  record_mixed_op(histograms ? histograms->ordered_iterate : nullptr, [&] {
    double sum = 0.0;
    for (auto &p : m.get<ByX>())
      sum += p.x;
    lb::do_not_optimize(sum);
  });
}

template <typename Map>
static void bmi_insert_mixed(MixedState<Map> &state,
                             MixedOpHistograms *histograms) {
  auto &m = *state.m;
  auto i = state.next_create_idx++;
  bool ok = false;
  record_mixed_op(histograms ? histograms->insert : nullptr, [&] {
    auto result =
        m.emplace(i + 1, generated_x(i), generated_y(i), generated_m(i));
    ok = result.second;
  });
  if (ok)
    state.live_ids.push_back(i + 1);
}

template <typename Map>
static void bmi_find_mixed(MixedState<Map> &state, size_t offset,
                           MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &idx = state.m->template get<ById>();
  uint64_t id = state.live_ids[(state.probe + offset) % state.live_ids.size()];
  record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
    auto it = idx.find(id);
    lb::do_not_optimize(it);
    if (it != idx.end())
      lb::do_not_optimize(it->x);
  });
}

template <typename Map>
static void bmi_find_hot_mixed(MixedState<Map> &state, size_t offset,
                               MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &idx_by_id = state.m->template get<ById>();
  size_t hot = std::max<size_t>(1, state.live_ids.size() / 10);
  size_t idx = (offset % 10 == 0)
                   ? (state.probe + offset) % state.live_ids.size()
                   : (state.probe + offset) % hot;
  uint64_t id = state.live_ids[idx];
  record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
    auto it = idx_by_id.find(id);
    lb::do_not_optimize(it);
    if (it != idx_by_id.end())
      lb::do_not_optimize(it->x);
  });
}

template <typename Map>
static void bmi_modify_mixed(MixedState<Map> &state, size_t offset, size_t salt,
                             MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &idx = state.m->template get<ById>();
  uint64_t id = state.live_ids[(state.probe + offset) % state.live_ids.size()];
  record_mixed_op(histograms ? histograms->modify : nullptr, [&] {
    auto it = idx.find(id);
    if (it != idx.end()) {
      idx.modify(it, [salt](Particle &p) { p.m = mixed_mass_probe(salt); });
    }
  });
}

template <typename Map>
static void bmi_remove_back_mixed(MixedState<Map> &state,
                                  MixedOpHistograms *histograms) {
  if (state.live_ids.empty())
    return;
  auto &idx = state.m->template get<ById>();
  uint64_t id = state.live_ids.back();
  state.live_ids.pop_back();
  record_mixed_op(histograms ? histograms->remove : nullptr, [&] {
    auto it = idx.find(id);
    if (it != idx.end())
      idx.erase(it);
  });
}

template <typename Map>
static void bmi_mass_range_mixed(MixedState<Map> &state, size_t offset,
                                 MixedOpHistograms *histograms) {
  auto &idx = state.m->template get<ByM>();
  double mass = mixed_mass_probe(state.probe + offset);
  record_mixed_op(histograms ? histograms->mass_range : nullptr, [&] {
    auto [beg, end] = idx.equal_range(mass);
    size_t count = 0;
    for (auto it = beg; it != end; ++it)
      ++count;
    lb::do_not_optimize(count);
  });
}

template <typename Map>
static void bmi_ordered_iterate_mixed(MixedState<Map> &state,
                                      MixedOpHistograms *histograms) {
  auto &idx = state.m->template get<ByX>();
  record_mixed_op(histograms ? histograms->ordered_iterate : nullptr, [&] {
    double sum = 0.0;
    for (auto &p : idx)
      sum += p.x;
    lb::do_not_optimize(sum);
  });
}

static void run_multimap_mixed(MixedState<ParticleMap> &state,
                               MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  auto &m = *state.m;
  for (size_t round = 0; round < kRounds; ++round) {
    for (size_t j = 0; j < kCreateBurst1 && state.next_create_idx < kN; ++j) {
      auto i = state.next_create_idx++;
      auto it = m.cend();
      record_mixed_op(histograms ? histograms->insert : nullptr, [&] {
        it =
            m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      });
      if (it != m.cend())
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst1 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
        auto it = m.find_primary(id);
        lb::do_not_optimize(it);
        if (it != m.cend())
          lb::do_not_optimize(it->x);
      });
    }
    state.probe += kFindBurst1;

    for (size_t j = 0; j < kModifyBurst1 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->modify : nullptr, [&] {
        auto it = m.find_primary(id);
        if (it != m.cend()) {
          m.modify<fastmm::ReindexOnly<2>>(*it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
          });
        }
      });
    }
    state.probe += kModifyBurst1;

    for (size_t j = 0; j < kRemoveBurst && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids.back();
      state.live_ids.pop_back();
      record_mixed_op(histograms ? histograms->remove : nullptr, [&] {
        auto it = m.find_primary(id);
        if (it != m.cend())
          m.remove(*it);
      });
    }

    for (size_t j = 0; j < kCreateBurst2 && state.next_create_idx < kN; ++j) {
      auto i = state.next_create_idx++;
      auto it = m.cend();
      record_mixed_op(histograms ? histograms->insert : nullptr, [&] {
        it =
            m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
      });
      if (it != m.cend())
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst2 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
        auto it = m.find_primary(id);
        lb::do_not_optimize(it);
        if (it != m.cend())
          lb::do_not_optimize(it->x);
      });
    }
    state.probe += kFindBurst2;

    for (size_t j = 0; j < kModifyBurst2 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->modify : nullptr, [&] {
        auto it = m.find_primary(id);
        if (it != m.cend()) {
          m.modify<fastmm::ReindexOnly<2>>(*it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
          });
        }
      });
    }
    state.probe += kModifyBurst2;
  }

  lb::do_not_optimize(m.size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_mixed(MixedState<Map> &state,
                          MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kRounds = 64;
  constexpr size_t kCreateBurst1 = 256;
  constexpr size_t kFindBurst1 = 512;
  constexpr size_t kModifyBurst1 = 512;
  constexpr size_t kRemoveBurst = 192;
  constexpr size_t kCreateBurst2 = 192;
  constexpr size_t kFindBurst2 = 256;
  constexpr size_t kModifyBurst2 = 256;

  auto &m = *state.m;
  auto &idx = m.template get<ById>();

  for (size_t round = 0; round < kRounds; ++round) {
    for (size_t j = 0; j < kCreateBurst1 && state.next_create_idx < kN; ++j) {
      auto i = state.next_create_idx++;
      bool ok = false;
      record_mixed_op(histograms ? histograms->insert : nullptr, [&] {
        auto result =
            m.emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        ok = result.second;
      });
      if (ok)
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst1 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
        auto it = idx.find(id);
        lb::do_not_optimize(it);
        if (it != idx.end())
          lb::do_not_optimize(it->x);
      });
    }
    state.probe += kFindBurst1;

    for (size_t j = 0; j < kModifyBurst1 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->modify : nullptr, [&] {
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 131 + j) % 10) + 1);
          });
        }
      });
    }
    state.probe += kModifyBurst1;

    for (size_t j = 0; j < kRemoveBurst && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids.back();
      state.live_ids.pop_back();
      record_mixed_op(histograms ? histograms->remove : nullptr, [&] {
        auto it = idx.find(id);
        if (it != idx.end())
          idx.erase(it);
      });
    }

    for (size_t j = 0; j < kCreateBurst2 && state.next_create_idx < kN; ++j) {
      auto i = state.next_create_idx++;
      bool ok = false;
      record_mixed_op(histograms ? histograms->insert : nullptr, [&] {
        auto result =
            m.emplace(kData.ids[i], kData.xs[i], kData.ys[i], kData.ms[i]);
        ok = result.second;
      });
      if (ok)
        state.live_ids.push_back(kData.ids[i]);
    }

    for (size_t j = 0; j < kFindBurst2 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->find_primary : nullptr, [&] {
        auto it = idx.find(id);
        lb::do_not_optimize(it);
        if (it != idx.end())
          lb::do_not_optimize(it->x);
      });
    }
    state.probe += kFindBurst2;

    for (size_t j = 0; j < kModifyBurst2 && !state.live_ids.empty(); ++j) {
      uint64_t id = state.live_ids[(state.probe + j) % state.live_ids.size()];
      record_mixed_op(histograms ? histograms->modify : nullptr, [&] {
        auto it = idx.find(id);
        if (it != idx.end()) {
          idx.modify(it, [round, j](Particle &p) {
            p.m = static_cast<double>(((round * 313 + j) % 10) + 1);
          });
        }
      });
    }
    state.probe += kModifyBurst2;
  }

  lb::do_not_optimize(m.size());
  lb::do_not_optimize(state.live_ids.size());
}

static void run_multimap_steady_churn(MixedState<ParticleMap> &state,
                                      MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kRounds = 4096;
  for (size_t round = 0; round < kRounds; ++round) {
    multimap_remove_back_mixed(state, histograms);
    multimap_insert_mixed(state, histograms);
    for (size_t j = 0; j < 8; ++j)
      multimap_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 4; ++j)
      multimap_modify_mixed(state, j + 8, round * 17 + j, histograms);
    state.probe += 12;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_steady_churn(MixedState<Map> &state,
                                 MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kRounds = 4096;
  for (size_t round = 0; round < kRounds; ++round) {
    bmi_remove_back_mixed(state, histograms);
    bmi_insert_mixed(state, histograms);
    for (size_t j = 0; j < 8; ++j)
      bmi_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 4; ++j)
      bmi_modify_mixed(state, j + 8, round * 17 + j, histograms);
    state.probe += 12;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

static void run_multimap_read_heavy(MixedState<ParticleMap> &state,
                                    MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kCycles = 512;
  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    for (size_t j = 0; j < 80; ++j)
      multimap_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 10; ++j)
      multimap_mass_range_mixed(state, j, histograms);
    for (size_t j = 0; j < 8; ++j)
      multimap_modify_mixed(state, j + 90, cycle * 29 + j, histograms);
    multimap_remove_back_mixed(state, histograms);
    multimap_insert_mixed(state, histograms);
    state.probe += 100;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_read_heavy(MixedState<Map> &state,
                               MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kCycles = 512;
  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    for (size_t j = 0; j < 80; ++j)
      bmi_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 10; ++j)
      bmi_mass_range_mixed(state, j, histograms);
    for (size_t j = 0; j < 8; ++j)
      bmi_modify_mixed(state, j + 90, cycle * 29 + j, histograms);
    bmi_remove_back_mixed(state, histograms);
    bmi_insert_mixed(state, histograms);
    state.probe += 100;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

static void
run_multimap_write_burst_read_burst(MixedState<ParticleMap> &state,
                                    MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kCycles = 16;
  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    for (size_t j = 0; j < 1000; ++j)
      multimap_insert_mixed(state, histograms);
    for (size_t j = 0; j < 2000; ++j)
      multimap_modify_mixed(state, j, cycle * 41 + j, histograms);
    for (size_t j = 0; j < 500; ++j)
      multimap_remove_back_mixed(state, histograms);
    for (size_t j = 0; j < 5000; ++j)
      multimap_find_mixed(state, j, histograms);
    multimap_ordered_iterate_mixed(state, histograms);
    state.probe += 5000;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void
run_bmi_write_burst_read_burst(MixedState<Map> &state,
                               MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kCycles = 16;
  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    for (size_t j = 0; j < 1000; ++j)
      bmi_insert_mixed(state, histograms);
    for (size_t j = 0; j < 2000; ++j)
      bmi_modify_mixed(state, j, cycle * 41 + j, histograms);
    for (size_t j = 0; j < 500; ++j)
      bmi_remove_back_mixed(state, histograms);
    for (size_t j = 0; j < 5000; ++j)
      bmi_find_mixed(state, j, histograms);
    bmi_ordered_iterate_mixed(state, histograms);
    state.probe += 5000;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

static void run_multimap_hot_key_skew(MixedState<ParticleMap> &state,
                                      MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kOps = 65536;
  for (size_t i = 0; i < kOps; ++i) {
    if (i % 10 == 8)
      multimap_modify_mixed(state, i, i, histograms);
    else if (i % 100 == 99) {
      multimap_remove_back_mixed(state, histograms);
      multimap_insert_mixed(state, histograms);
    } else
      multimap_find_hot_mixed(state, i, histograms);
    ++state.probe;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_hot_key_skew(MixedState<Map> &state,
                                 MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kOps = 65536;
  for (size_t i = 0; i < kOps; ++i) {
    if (i % 10 == 8)
      bmi_modify_mixed(state, i, i, histograms);
    else if (i % 100 == 99) {
      bmi_remove_back_mixed(state, histograms);
      bmi_insert_mixed(state, histograms);
    } else
      bmi_find_hot_mixed(state, i, histograms);
    ++state.probe;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

static void
run_multimap_range_query_mixed(MixedState<ParticleMap> &state,
                               MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kCycles = 1024;
  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    for (size_t j = 0; j < 10; ++j)
      multimap_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 4; ++j)
      multimap_mass_range_mixed(state, j, histograms);
    for (size_t j = 0; j < 4; ++j)
      multimap_modify_mixed(state, j + 14, cycle * 53 + j, histograms);
    multimap_remove_back_mixed(state, histograms);
    multimap_insert_mixed(state, histograms);
    state.probe += 20;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_range_query_mixed(MixedState<Map> &state,
                                      MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kCycles = 1024;
  for (size_t cycle = 0; cycle < kCycles; ++cycle) {
    for (size_t j = 0; j < 10; ++j)
      bmi_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 4; ++j)
      bmi_mass_range_mixed(state, j, histograms);
    for (size_t j = 0; j < 4; ++j)
      bmi_modify_mixed(state, j + 14, cycle * 53 + j, histograms);
    bmi_remove_back_mixed(state, histograms);
    bmi_insert_mixed(state, histograms);
    state.probe += 20;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

static void run_multimap_growing_load(MixedState<ParticleMap> &state,
                                      MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kSteps = kN / 2;
  for (size_t step = 0; step < kSteps; ++step) {
    multimap_insert_mixed(state, histograms);
    for (size_t j = 0; j < 4; ++j)
      multimap_find_mixed(state, j, histograms);
    if (step % 2 == 0)
      multimap_modify_mixed(state, step, step, histograms);
    if (step % 16 == 0)
      multimap_mass_range_mixed(state, step, histograms);
    state.probe += 4;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void run_bmi_growing_load(MixedState<Map> &state,
                                 MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kSteps = kN / 2;
  for (size_t step = 0; step < kSteps; ++step) {
    bmi_insert_mixed(state, histograms);
    for (size_t j = 0; j < 4; ++j)
      bmi_find_mixed(state, j, histograms);
    if (step % 2 == 0)
      bmi_modify_mixed(state, step, step, histograms);
    if (step % 16 == 0)
      bmi_mass_range_mixed(state, step, histograms);
    state.probe += 4;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

static void
run_multimap_near_capacity_churn(MixedState<ParticleMap> &state,
                                 MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kRounds = 2048;
  for (size_t round = 0; round < kRounds; ++round) {
    multimap_remove_back_mixed(state, histograms);
    multimap_insert_mixed(state, histograms);
    for (size_t j = 0; j < 10; ++j)
      multimap_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 5; ++j)
      multimap_modify_mixed(state, j + 10, round * 67 + j, histograms);
    state.probe += 15;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

template <typename Map>
static void
run_bmi_near_capacity_churn(MixedState<Map> &state,
                            MixedOpHistograms *histograms = nullptr) {
  constexpr size_t kRounds = 2048;
  for (size_t round = 0; round < kRounds; ++round) {
    bmi_remove_back_mixed(state, histograms);
    bmi_insert_mixed(state, histograms);
    for (size_t j = 0; j < 10; ++j)
      bmi_find_mixed(state, j, histograms);
    for (size_t j = 0; j < 5; ++j)
      bmi_modify_mixed(state, j + 10, round * 67 + j, histograms);
    state.probe += 15;
  }
  lb::do_not_optimize(state.m->size());
  lb::do_not_optimize(state.live_ids.size());
}

// ---------------------------------------------------------------------------
// Hot/Cold Entity Mixed benchmark
// Maintains N entities, α=20% hot (in BySeq + ByM), rest cold (primary only).
// Each cycle: transition hot↔cold, update hot, query buckets, churn entities.
// FastMM: native unindex<>/index<> — no data copy, O(1) primary lookup.
// BMI:    HotOnlyBMI (3 indices) for hot + unordered_map for cold — copies on
//         transition, modify reindexes all 3 hot indices unconditionally.
// ---------------------------------------------------------------------------
static constexpr size_t kHotN = kN / 5; // 20% hot
static constexpr size_t kHCTransPerCycle = 4;
static constexpr size_t kHCUpdatePerCycle = 32;
static constexpr size_t kHCQueryPerCycle = 4;
static constexpr size_t kHCChurnPerCycle = 2;
static constexpr size_t kHCCycles = 1024;

static double hot_cold_mass(size_t i) {
  return static_cast<double>(i % 10 + 1);
}

// State: all_ids[0..hot_count) are hot; [hot_count..) are cold.
// Cooling removes from the back of the hot section.
// Warming promotes from the front of the cold section.
// A single swap per transition keeps the invariant without a separate list.
struct HotColdMMState {
  std::unique_ptr<HotColdParticleMap> m;
  std::vector<uint64_t> all_ids;
  size_t hot_count{kHotN};
  uint64_t next_id{kN + 1};
  size_t probe{};
};

template <typename HotBMI>
struct HotColdBMIState {
  std::unordered_map<uint64_t, Particle> cold_map;
  std::unique_ptr<HotBMI> hot_bmi;
  std::vector<uint64_t> all_ids;
  size_t hot_count{kHotN};
  uint64_t next_id{kN + 1};
  size_t probe{};
};

static HotColdMMState make_hot_cold_mm_state() {
  HotColdMMState st;
  st.m = std::make_unique<HotColdParticleMap>();
  st.all_ids.reserve(kN);
  for (size_t i = 0; i < kN; ++i) {
    auto it = st.m->insert<false>(i + 1, generated_x(i), generated_y(i),
                                  hot_cold_mass(i));
    if (it == st.m->cend())
      std::abort();
    st.all_ids.push_back(i + 1);
  }
  for (size_t i = 0; i < kHotN; ++i) {
    auto it = st.m->find_primary(st.all_ids[i]);
    st.m->index<BySeq>(*it);
    st.m->index<ByM>(*it);
  }
  return st;
}

template <typename HotBMI>
static HotColdBMIState<HotBMI> make_hot_cold_bmi_state() {
  HotColdBMIState<HotBMI> st;
  st.hot_bmi = std::make_unique<HotBMI>();
  st.hot_bmi->reserve(kHotN);
  st.all_ids.reserve(kN);
  st.cold_map.reserve(kN - kHotN);
  for (size_t i = 0; i < kHotN; ++i) {
    st.hot_bmi->emplace(i + 1, generated_x(i), generated_y(i),
                        hot_cold_mass(i));
    st.all_ids.push_back(i + 1);
  }
  for (size_t i = kHotN; i < kN; ++i) {
    st.cold_map.try_emplace(i + 1, i + 1, generated_x(i), generated_y(i),
                            hot_cold_mass(i));
    st.all_ids.push_back(i + 1);
  }
  return st;
}

static void run_multimap_hot_cold(HotColdMMState &st,
                                  HotColdOpHistograms *hists = nullptr) {
  auto &m = *st.m;
  for (size_t cycle = 0; cycle < kHCCycles; ++cycle) {
    // 1. Cool kHCTransPerCycle hot entities (back of hot section → front of cold)
    size_t n_cool = std::min(kHCTransPerCycle, st.hot_count);
    for (size_t t = 0; t < n_cool; ++t) {
      uint64_t id = st.all_ids[st.hot_count - 1 - t];
      record_mixed_op(hists ? hists->transition_out : nullptr, [&] {
        auto it = m.find_primary(id);
        if (it != m.cend()) {
          m.unindex<BySeq>(*it);
          m.unindex<ByM>(*it);
        }
      });
    }
    // 2. Warm kHCTransPerCycle cold entities (front of cold section → back of hot)
    size_t n_warm =
        std::min(kHCTransPerCycle, st.all_ids.size() - st.hot_count);
    for (size_t t = 0; t < n_warm; ++t) {
      uint64_t id = st.all_ids[st.hot_count + t];
      record_mixed_op(hists ? hists->transition_in : nullptr, [&] {
        auto it = m.find_primary(id);
        if (it != m.cend()) {
          m.index<BySeq>(*it);
          m.index<ByM>(*it);
        }
      });
    }
    // Swap cooled (tail of hot) with warmed (head of cold) to keep invariant
    size_t n_swap = std::min(n_cool, n_warm);
    for (size_t t = 0; t < n_swap; ++t)
      std::swap(st.all_ids[st.hot_count - n_cool + t],
                st.all_ids[st.hot_count + t]);

    // 3. Update kHCUpdatePerCycle hot entities via BySeq (active list iteration).
    //    FastMM: modify<ByM> reindexes only the bucket tree, skipping BySeq/id.
    //    BMI:    modify reindexes all 3 hot indices even for unchanged seq/id keys.
    {
      auto &seq = m.get<BySeq>();
      auto seq_it = seq.begin();
      for (size_t u = 0; u < kHCUpdatePerCycle && seq_it != seq.end();
           ++u, ++seq_it) {
        record_mixed_op(hists ? hists->update_hot : nullptr, [&, cycle, u] {
          m.modify<fastmm::ReindexOnlyByTag<ByM>>(
              *seq_it, [cycle, u](Particle &p) {
                p.m = static_cast<double>((cycle * 7 + u) % 10 + 1);
              });
        });
      }
    }

    // 4. Query kHCQueryPerCycle buckets (hot ByM index only)
    for (size_t q = 0; q < kHCQueryPerCycle; ++q) {
      double mass = static_cast<double>((st.probe + q) % 10 + 1);
      record_mixed_op(hists ? hists->query_bucket : nullptr, [&] {
        auto &idx = m.get<ByM>();
        auto [beg, end] = idx.equal_range(mass);
        size_t count = 0;
        for (auto it = beg; it != end; ++it)
          ++count;
        lb::do_not_optimize(count);
      });
    }

    // 5. Churn: remove cold entity, insert new cold entity
    for (size_t c = 0; c < kHCChurnPerCycle; ++c) {
      if (st.all_ids.size() > st.hot_count) {
        uint64_t id = st.all_ids.back();
        st.all_ids.pop_back();
        record_mixed_op(hists ? hists->churn_remove : nullptr,
                        [&] { m.remove(id); });
      }
      uint64_t new_id = st.next_id++;
      record_mixed_op(hists ? hists->churn_insert : nullptr, [&] {
        auto it = m.insert<false>(new_id, generated_x(new_id % kN),
                                  generated_y(new_id % kN),
                                  hot_cold_mass(new_id));
        if (it != m.cend())
          st.all_ids.push_back(new_id);
      });
    }

    st.probe += kHCUpdatePerCycle;
  }
  lb::do_not_optimize(m.size());
  lb::do_not_optimize(st.all_ids.size());
}

template <typename HotBMI>
static void run_bmi_hot_cold(HotColdBMIState<HotBMI> &st,
                             HotColdOpHistograms *hists = nullptr) {
  auto &hot = *st.hot_bmi;
  auto &hot_idx = hot.template get<ById>();

  for (size_t cycle = 0; cycle < kHCCycles; ++cycle) {
    // 1. Cool kHCTransPerCycle hot entities (erase from hot_bmi, store in cold_map)
    size_t n_cool = std::min(kHCTransPerCycle, st.hot_count);
    for (size_t t = 0; t < n_cool; ++t) {
      uint64_t id = st.all_ids[st.hot_count - 1 - t];
      record_mixed_op(hists ? hists->transition_out : nullptr, [&] {
        auto it = hot_idx.find(id);
        if (it != hot_idx.end()) {
          st.cold_map.try_emplace(id, *it);
          hot_idx.erase(it);
        }
      });
    }
    // 2. Warm kHCTransPerCycle cold entities (move from cold_map to hot_bmi)
    size_t n_warm =
        std::min(kHCTransPerCycle, st.all_ids.size() - st.hot_count);
    for (size_t t = 0; t < n_warm; ++t) {
      uint64_t id = st.all_ids[st.hot_count + t];
      record_mixed_op(hists ? hists->transition_in : nullptr, [&] {
        auto cm_it = st.cold_map.find(id);
        if (cm_it != st.cold_map.end()) {
          hot.emplace(cm_it->second);
          st.cold_map.erase(cm_it);
        }
      });
    }
    size_t n_swap = std::min(n_cool, n_warm);
    for (size_t t = 0; t < n_swap; ++t)
      std::swap(st.all_ids[st.hot_count - n_cool + t],
                st.all_ids[st.hot_count + t]);

    // 3. Update kHCUpdatePerCycle hot entities via BySeq iteration.
    //    BMI modify reindexes all 3 hot indices (ByID, BySeq, ByM) even though
    //    only the ByM position may change.
    {
      auto &seq = hot.template get<BySeq>();
      auto seq_it = seq.begin();
      for (size_t u = 0; u < kHCUpdatePerCycle && seq_it != seq.end();
           ++u) {
        auto cur = seq_it++;
        record_mixed_op(hists ? hists->update_hot : nullptr, [&, cycle, u] {
          seq.modify(cur, [cycle, u](Particle &p) {
            p.m = static_cast<double>((cycle * 7 + u) % 10 + 1);
          });
        });
      }
    }

    // 4. Query kHCQueryPerCycle buckets (hot ByM index only)
    for (size_t q = 0; q < kHCQueryPerCycle; ++q) {
      double mass = static_cast<double>((st.probe + q) % 10 + 1);
      record_mixed_op(hists ? hists->query_bucket : nullptr, [&] {
        auto &idx = hot.template get<ByM>();
        auto [beg, end] = idx.equal_range(mass);
        size_t count = 0;
        for (auto it = beg; it != end; ++it)
          ++count;
        lb::do_not_optimize(count);
      });
    }

    // 5. Churn: remove cold entity, insert new cold entity
    for (size_t c = 0; c < kHCChurnPerCycle; ++c) {
      if (st.all_ids.size() > st.hot_count) {
        uint64_t id = st.all_ids.back();
        st.all_ids.pop_back();
        record_mixed_op(hists ? hists->churn_remove : nullptr,
                        [&] { st.cold_map.erase(id); });
      }
      uint64_t new_id = st.next_id++;
      record_mixed_op(hists ? hists->churn_insert : nullptr, [&] {
        auto [it, ok] = st.cold_map.try_emplace(
            new_id, new_id, generated_x(new_id % kN), generated_y(new_id % kN),
            hot_cold_mass(new_id));
        if (ok)
          st.all_ids.push_back(new_id);
      });
    }

    st.probe += kHCUpdatePerCycle;
  }
  lb::do_not_optimize(hot.size());
  lb::do_not_optimize(st.cold_map.size());
  lb::do_not_optimize(st.all_ids.size());
}

static auto make_multimap_mixed_state_with_initial(size_t initial) {
  return make_mixed_state<ParticleMap>(
      initial, [] { return std::make_unique<ParticleMap>(); },
      [](ParticleMap &m, size_t i) {
        auto it = m.insert<true>(i + 1, generated_x(i), generated_y(i),
                                 generated_m(i));
        if (it == m.cend())
          std::abort();
      });
}

static auto make_default_bmi_mixed_state_with_initial(size_t initial) {
  return make_mixed_state<ParticleBMI>(
      initial,
      [] {
        auto m = std::make_unique<ParticleBMI>();
        m->reserve(kN);
        return m;
      },
      [](ParticleBMI &m, size_t i) {
        auto [it, ok] =
            m.emplace(i + 1, generated_x(i), generated_y(i), generated_m(i));
        if (!ok)
          std::abort();
      });
}

static auto make_pool_bmi_mixed_state_with_initial(size_t initial) {
  return make_mixed_state<ParticleBMIPool>(
      initial,
      [] {
        auto m = std::make_unique<ParticleBMIPool>();
        m->reserve(kN);
        return m;
      },
      [](ParticleBMIPool &m, size_t i) {
        auto [it, ok] =
            m.emplace(i + 1, generated_x(i), generated_y(i), generated_m(i));
        if (!ok)
          std::abort();
      });
}

static void register_mixed_benches() {
  auto cfg = container_op_cfg();
  auto extended_cfg = extended_mixed_cfg();
  auto register_workload = [&](const char *suffix, size_t initial, auto run_mm,
                               auto run_bmi) {
    std::string mm_name = std::string{"MultiMap_"} + suffix;
    std::string default_name = std::string{"DefaultBMI_"} + suffix;
    std::string pool_name = std::string{"PoolBMI_"} + suffix;

    register_latency_bench(
        mm_name.c_str(), extended_cfg,
        [initial] { return make_multimap_mixed_state_with_initial(initial); },
        [run_mm](auto &state) mutable { run_mm(state); });
    register_mixed_op_latency_bench(
        mm_name.c_str(), extended_cfg,
        [initial] { return make_multimap_mixed_state_with_initial(initial); },
        [run_mm](auto &state, MixedOpHistograms *histograms) mutable {
          run_mm(state, histograms);
        });

    register_latency_bench(
        default_name.c_str(), extended_cfg,
        [initial] {
          return make_default_bmi_mixed_state_with_initial(initial);
        },
        [run_bmi](auto &state) mutable { run_bmi(state); });
    register_mixed_op_latency_bench(
        default_name.c_str(), extended_cfg,
        [initial] {
          return make_default_bmi_mixed_state_with_initial(initial);
        },
        [run_bmi](auto &state, MixedOpHistograms *histograms) mutable {
          run_bmi(state, histograms);
        });

    register_latency_bench(
        pool_name.c_str(), extended_cfg,
        [initial] { return make_pool_bmi_mixed_state_with_initial(initial); },
        [run_bmi](auto &state) mutable { run_bmi(state); });
    register_mixed_op_latency_bench(
        pool_name.c_str(), extended_cfg,
        [initial] { return make_pool_bmi_mixed_state_with_initial(initial); },
        [run_bmi](auto &state, MixedOpHistograms *histograms) mutable {
          run_bmi(state, histograms);
        });
  };

  register_latency_bench(
      "MultiMap_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleMap>(
            [] { return std::make_unique<ParticleMap>(); },
            [](ParticleMap &m, size_t i) {
              auto it = m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                       kData.ms[i]);
              if (it == m.cend())
                std::abort();
            });
      },
      [](auto &state) { run_multimap_mixed(state); });

  register_mixed_op_latency_bench(
      "MultiMap_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleMap>(
            [] { return std::make_unique<ParticleMap>(); },
            [](ParticleMap &m, size_t i) {
              auto it = m.insert<true>(kData.ids[i], kData.xs[i], kData.ys[i],
                                       kData.ms[i]);
              if (it == m.cend())
                std::abort();
            });
      },
      [](auto &state, MixedOpHistograms *histograms) {
        run_multimap_mixed(state, histograms);
      });

  register_latency_bench(
      "DefaultBMI_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleBMI>(
            [] {
              auto m = std::make_unique<ParticleBMI>();
              m->reserve(kN);
              return m;
            },
            [](ParticleBMI &m, size_t i) {
              auto [it, ok] = m.emplace(kData.ids[i], kData.xs[i], kData.ys[i],
                                        kData.ms[i]);
              if (!ok)
                std::abort();
            });
      },
      [](auto &state) { run_bmi_mixed(state); });

  register_mixed_op_latency_bench(
      "DefaultBMI_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleBMI>(
            [] {
              auto m = std::make_unique<ParticleBMI>();
              m->reserve(kN);
              return m;
            },
            [](ParticleBMI &m, size_t i) {
              auto [it, ok] = m.emplace(kData.ids[i], kData.xs[i], kData.ys[i],
                                        kData.ms[i]);
              if (!ok)
                std::abort();
            });
      },
      [](auto &state, MixedOpHistograms *histograms) {
        run_bmi_mixed(state, histograms);
      });

  register_latency_bench(
      "PoolBMI_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleBMIPool>(
            [] {
              auto m = std::make_unique<ParticleBMIPool>();
              m->reserve(kN);
              return m;
            },
            [](ParticleBMIPool &m, size_t i) {
              auto [it, ok] = m.emplace(kData.ids[i], kData.xs[i], kData.ys[i],
                                        kData.ms[i]);
              if (!ok)
                std::abort();
            });
      },
      [](auto &state) { run_bmi_mixed(state); });

  register_mixed_op_latency_bench(
      "PoolBMI_Mixed", cfg,
      [] {
        return make_mixed_state<ParticleBMIPool>(
            [] {
              auto m = std::make_unique<ParticleBMIPool>();
              m->reserve(kN);
              return m;
            },
            [](ParticleBMIPool &m, size_t i) {
              auto [it, ok] = m.emplace(kData.ids[i], kData.xs[i], kData.ys[i],
                                        kData.ms[i]);
              if (!ok)
                std::abort();
            });
      },
      [](auto &state, MixedOpHistograms *histograms) {
        run_bmi_mixed(state, histograms);
      });

  register_workload(
      "SteadyChurn", kN / 2,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_steady_churn(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_steady_churn(state, histograms);
      });
  register_workload(
      "ReadHeavyMixed", kN / 2,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_read_heavy(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_read_heavy(state, histograms);
      });
  register_workload(
      "WriteBurstReadBurst", kN / 2,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_write_burst_read_burst(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_write_burst_read_burst(state, histograms);
      });
  register_workload(
      "HotKeySkewMixed", kN / 2,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_hot_key_skew(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_hot_key_skew(state, histograms);
      });
  register_workload(
      "RangeQueryMixed", kN / 2,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_range_query_mixed(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_range_query_mixed(state, histograms);
      });
  register_workload(
      "GrowingLoadMixed", 0,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_growing_load(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_growing_load(state, histograms);
      });
  register_workload(
      "NearCapacityChurn", kN * 95 / 100,
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_multimap_near_capacity_churn(state, histograms);
      },
      [](auto &state, MixedOpHistograms *histograms = nullptr) {
        run_bmi_near_capacity_churn(state, histograms);
      });
}

// ---------------------------------------------------------------------------
// Hot/Cold benchmark registration
// ---------------------------------------------------------------------------
static lb::BenchConfig hot_cold_cfg() {
  lb::BenchConfig cfg;
  cfg.warmup_iters = 2;
  cfg.measure_iters = 20;
  apply_iteration_override(cfg);
  return cfg;
}

template <typename Setup, typename Fn>
static void register_hot_cold_latency_bench(const char *base_name,
                                             lb::BenchConfig cfg,
                                             Setup &&setup, Fn &&fn) {
  const std::string base{base_name};
  std::vector<std::string> out_names{
      base + "_HotCold_TransIn",     base + "_HotCold_TransOut",
      base + "_HotCold_UpdateHot",   base + "_HotCold_QueryBucket",
      base + "_HotCold_ChurnInsert", base + "_HotCold_ChurnRemove"};

  registry().push_back(
      {base + "_HotCold", out_names,
       [out_names, cfg, setup = std::forward<Setup>(setup),
        fn = std::forward<Fn>(fn)]() mutable {
         HotColdOpHistograms hists(cfg);

         for (int64_t i = 0; i < cfg.warmup_iters; ++i) {
           auto state = setup();
           lb::clobber_memory();
           fn(state, nullptr);
           lb::clobber_memory();
         }

         hists.reset();
         for (int64_t i = 0; i < cfg.measure_iters; ++i) {
           auto state = setup();
           lb::clobber_memory();
           fn(state, &hists);
           lb::clobber_memory();
         }

         std::vector<DetailedReport> reports;
         auto add = [&](const std::string &name, hdr_histogram *h) {
           if (h->total_count > 0)
             reports.push_back(DetailedReport::from(name, h));
         };
         add(out_names[0], hists.transition_in);
         add(out_names[1], hists.transition_out);
         add(out_names[2], hists.update_hot);
         add(out_names[3], hists.query_bucket);
         add(out_names[4], hists.churn_insert);
         add(out_names[5], hists.churn_remove);
         return reports;
       }});
}

static void register_hot_cold_benches() {
  auto cfg = hot_cold_cfg();

  register_hot_cold_latency_bench(
      "MultiMap", cfg, [] { return make_hot_cold_mm_state(); },
      [](HotColdMMState &st, HotColdOpHistograms *h) {
        run_multimap_hot_cold(st, h);
      });

  register_hot_cold_latency_bench(
      "DefaultBMI", cfg,
      [] { return make_hot_cold_bmi_state<HotOnlyBMI>(); },
      [](HotColdBMIState<HotOnlyBMI> &st, HotColdOpHistograms *h) {
        run_bmi_hot_cold(st, h);
      });

  register_hot_cold_latency_bench(
      "PoolBMI", cfg,
      [] { return make_hot_cold_bmi_state<HotOnlyBMIPool>(); },
      [](HotColdBMIState<HotOnlyBMIPool> &st, HotColdOpHistograms *h) {
        run_bmi_hot_cold(st, h);
      });
}

static void register_all_benches() {
  const char *const create[] = {"MultiMap_Create", "DefaultBMI_Create",
                                "PoolBMI_Create"};
  const char *const find_primary[] = {
      "MultiMap_FindPrimary", "DefaultBMI_FindPrimary", "PoolBMI_FindPrimary"};
  const char *const remove[] = {"MultiMap_Remove", "DefaultBMI_Remove",
                                "PoolBMI_Remove"};
  const char *const bulk_iterate[] = {
      "MultiMap_BulkIterate", "DefaultBMI_BulkIterate", "PoolBMI_BulkIterate"};
  const char *const mass_range[] = {
      "MultiMap_MassRange", "DefaultBMI_MassRange", "PoolBMI_MassRange"};
  const char *const ordered_iterate[] = {"MultiMap_OrderedIterate",
                                         "DefaultBMI_OrderedIterate",
                                         "PoolBMI_OrderedIterate"};
  const char *const modify[] = {"MultiMap_Modify", "DefaultBMI_Modify",
                                "PoolBMI_Modify"};
  const char *const level_walk[] = {
      "MultiMap_LevelWalk", "DefaultBMI_LevelWalk", "PoolBMI_LevelWalk"};
  const char *const mixed[] = {"MultiMap_Mixed",
                               "DefaultBMI_Mixed",
                               "PoolBMI_Mixed",
                               "MultiMap_SteadyChurn",
                               "DefaultBMI_SteadyChurn",
                               "PoolBMI_SteadyChurn",
                               "MultiMap_ReadHeavyMixed",
                               "DefaultBMI_ReadHeavyMixed",
                               "PoolBMI_ReadHeavyMixed",
                               "MultiMap_WriteBurstReadBurst",
                               "DefaultBMI_WriteBurstReadBurst",
                               "PoolBMI_WriteBurstReadBurst",
                               "MultiMap_HotKeySkewMixed",
                               "DefaultBMI_HotKeySkewMixed",
                               "PoolBMI_HotKeySkewMixed",
                               "MultiMap_RangeQueryMixed",
                               "DefaultBMI_RangeQueryMixed",
                               "PoolBMI_RangeQueryMixed",
                               "MultiMap_GrowingLoadMixed",
                               "DefaultBMI_GrowingLoadMixed",
                               "PoolBMI_GrowingLoadMixed",
                               "MultiMap_NearCapacityChurn",
                               "DefaultBMI_NearCapacityChurn",
                               "PoolBMI_NearCapacityChurn",
                               "MultiMap_Mixed_Insert",
                               "MultiMap_Mixed_FindPrimary",
                               "MultiMap_Mixed_Modify",
                               "MultiMap_Mixed_Remove",
                               "DefaultBMI_Mixed_Insert",
                               "DefaultBMI_Mixed_FindPrimary",
                               "DefaultBMI_Mixed_Modify",
                               "DefaultBMI_Mixed_Remove",
                               "PoolBMI_Mixed_Insert",
                               "PoolBMI_Mixed_FindPrimary",
                               "PoolBMI_Mixed_Modify",
                               "PoolBMI_Mixed_Remove"};
  const char *const hot_cold[] = {
      "MultiMap_HotCold",                 "DefaultBMI_HotCold",
      "PoolBMI_HotCold",                  "MultiMap_HotCold_TransIn",
      "MultiMap_HotCold_TransOut",        "MultiMap_HotCold_UpdateHot",
      "MultiMap_HotCold_QueryBucket",     "MultiMap_HotCold_ChurnInsert",
      "MultiMap_HotCold_ChurnRemove",     "DefaultBMI_HotCold_TransIn",
      "DefaultBMI_HotCold_TransOut",      "DefaultBMI_HotCold_UpdateHot",
      "DefaultBMI_HotCold_QueryBucket",   "DefaultBMI_HotCold_ChurnInsert",
      "DefaultBMI_HotCold_ChurnRemove",   "PoolBMI_HotCold_TransIn",
      "PoolBMI_HotCold_TransOut",         "PoolBMI_HotCold_UpdateHot",
      "PoolBMI_HotCold_QueryBucket",      "PoolBMI_HotCold_ChurnInsert",
      "PoolBMI_HotCold_ChurnRemove"};

  if (should_register_any(create))
    register_create_benches();
  if (should_register_any(find_primary))
    register_find_primary_benches();
  if (should_register_any(remove))
    register_remove_benches();
  if (should_register_any(bulk_iterate))
    register_bulk_iterate_benches();
  if (should_register_any(mass_range))
    register_mass_range_benches();
  if (should_register_any(ordered_iterate))
    register_ordered_iterate_benches();
  if (should_register_any(modify))
    register_modify_benches();
  if (should_register_any(level_walk))
    register_level_walk_benches();
  if (should_register_any(mixed))
    register_mixed_benches();
  if (should_register_any(hot_cold))
    register_hot_cold_benches();
}

static void print_gbench_table(std::ostream &os,
                               const std::vector<DetailedReport> &reports,
                               bool export_histogram,
                               const std::string &hist_path) {
  // Timestamp
  std::time_t now = std::time(nullptr);
  char tbuf[64];
  std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S+00:00",
                std::gmtime(&now));
  os << tbuf << "\n";

  // System info
#ifdef __linux__
  {
    long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    std::string cpu_model;
    double cpu_mhz = 0.0;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (cpu_model.empty() && line.rfind("model name", 0) == 0) {
        auto pos = line.find(':');
        if (pos != std::string::npos)
          cpu_model = line.substr(pos + 2);
      }
      if (cpu_mhz == 0.0 && line.rfind("cpu MHz", 0) == 0) {
        auto pos = line.find(':');
        if (pos != std::string::npos)
          cpu_mhz = std::stod(line.substr(pos + 2));
      }
    }
    os << "Run on (" << ncpus << " X " << std::fixed << std::setprecision(2)
       << cpu_mhz << " MHz CPU s)\n";
    if (!cpu_model.empty())
      os << "CPU: " << cpu_model << "\n";
  }
#endif

  // Compute name column width from actual benchmark names
  int name_w = 32;
  for (const auto &r : reports) {
    int n = static_cast<int>(r.summary.name.size()) + 2;
    if (n > name_w)
      name_w = n;
  }

  constexpr int kTimeW = 14;
  constexpr int kThputW = 16;
  constexpr int kItersW = 11;
  int total_w = name_w + 6 * kTimeW + kThputW + kItersW;
  std::string sep(total_w, '-');

  os << sep << "\n";
  os << std::left << std::setw(name_w) << "Benchmark" << std::right
     << std::setw(kTimeW) << "Mean" << std::setw(kTimeW) << "p50"
     << std::setw(kTimeW) << "p90" << std::setw(kTimeW) << "p99"
     << std::setw(kTimeW) << "p99.9" << std::setw(kTimeW) << "Max"
     << std::setw(kThputW) << "Throughput"
     << std::setw(kItersW) << "Iters"
     << "\n";
  os << sep << "\n";

  for (const auto &r : reports)
    r.summary.print_gbench(os, name_w, kTimeW, kThputW, kItersW);

  os << sep << "\n";

  if (export_histogram)
    os << "histogram_json=" << hist_path << "\n";
}

static int64_t parse_i64_at_least(const char *arg, const char *option,
                                  int64_t min_value) {
  char *end = nullptr;
  errno = 0;
  long long value = std::strtoll(arg, &end, 10);
  if (errno != 0 || end == arg || *end != '\0' || value < min_value) {
    std::cerr << "invalid " << option << " value: " << arg << "\n";
    std::exit(2);
  }
  return static_cast<int64_t>(value);
}

int main(int argc, char **argv) {
  bool json = false;
  bool export_histogram_json = true;
  bool pin_core = true;
  int core = 0;
  std::string filter;
  std::string histogram_json_path = "particle_latency_histograms.json";

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!std::strcmp(arg, "--json"))
      json = true;
    else if (!std::strcmp(arg, "--hist-json") && i + 1 < argc)
      histogram_json_path = argv[++i];
    else if (!std::strncmp(arg, "--hist-json=", 12))
      histogram_json_path = arg + 12;
    else if (!std::strcmp(arg, "--no-hist-json"))
      export_histogram_json = false;
    else if (!std::strcmp(arg, "--core") && i + 1 < argc)
      core = static_cast<int>(parse_i64_at_least(argv[++i], "--core", 0));
    else if (!std::strncmp(arg, "--core=", 7))
      core = static_cast<int>(parse_i64_at_least(arg + 7, "--core", 0));
    else if (!std::strcmp(arg, "--no-pin"))
      pin_core = false;
    else if (!std::strcmp(arg, "--iterations") && i + 1 < argc)
      g_measure_iters_override =
          parse_i64_at_least(argv[++i], "--iterations", 1);
    else if (!std::strncmp(arg, "--iterations=", 13))
      g_measure_iters_override =
          parse_i64_at_least(arg + 13, "--iterations", 1);
    else if (!std::strcmp(arg, "--filter") && i + 1 < argc)
      filter = argv[++i];
    else if (!std::strncmp(arg, "--filter=", 9))
      filter = arg + 9;
    else {
      std::cerr << "unknown or incomplete option: " << arg << "\n";
      return 2;
    }
  }

  if (pin_core)
    pin_to_core(core);
  g_filter = filter;
  register_all_benches();

  std::vector<DetailedReport> reports;
  for (auto &bench : registry()) {
    if (!should_run(bench, filter))
      continue;
    std::cerr << "running " << bench.name << "...\n";
    auto bench_reports = bench.run();
    for (auto &report : bench_reports) {
      if (should_keep_report(report, filter))
        reports.push_back(std::move(report));
    }
  }

  if (export_histogram_json)
    write_histogram_json_file(histogram_json_path, reports);

  if (json) {
    std::cout << "[";
    for (size_t i = 0; i < reports.size(); ++i) {
      if (i)
        std::cout << ",";
      reports[i].summary.print_json(std::cout);
    }
    std::cout << "]\n";
  } else {
    print_gbench_table(std::cout, reports, export_histogram_json,
                       histogram_json_path);
  }

  return 0;
}
