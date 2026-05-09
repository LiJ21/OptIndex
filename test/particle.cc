#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "optindex.h"

struct Particle {
  uint64_t id;
  double x;
  double y;
  double m;  // mass — can repeat

  Particle(uint64_t id, double x, double y, double m)
      : id(id), x(x), y(y), m(m) {}
};

struct Idgetter {
  using type = uint64_t;
  const uint64_t &operator()(const Particle &p) const { return p.id; }
};
struct Xgetter {
  using type = double;
  const double &operator()(const Particle &p) const { return p.x; }
};
struct Ygetter {
  using type = double;
  const double &operator()(const Particle &p) const { return p.y; }
};
struct Mgetter {
  using type = double;
  const double &operator()(const Particle &p) const { return p.m; }
};

struct IdHash {
  std::size_t operator()(uint64_t id) const {
    return std::hash<uint64_t>{}(id);
  }
};
struct IdEqual {
  bool operator()(uint64_t a, uint64_t b) const { return a == b; }
};

// Tags
struct ById {};
struct ByX  {};
struct ByY  {};
struct ByM  {};
struct BySeq{};

static constexpr std::size_t kMaxParticles = 64;

using ParticleMap = optindex::FixedSizeOptIndex<
    Particle, kMaxParticles,
    optindex::Unordered<Idgetter, IdHash, IdEqual, 128>,
    optindex::IndexBy<optindex::Ordered<Xgetter, std::less<double>>,        ByX>,
    optindex::IndexBy<optindex::Ordered<Ygetter, std::less<double>>,        ByY>,
    optindex::IndexBy<optindex::OrderedNonUnique<Mgetter, std::less<double>>, ByM>,
    optindex::IndexBy<optindex::List,                                        BySeq>
>;

using Slot = ParticleMap::SlotType;

// Helper: create + insert into all secondaries (ByX, ByY, ByM, BySeq).
static auto InsertAll(ParticleMap &m, uint64_t id, double x, double y,
                      double mass) {
  return m.create_all(id, x, y, mass);
}

class ParticleMapTest : public ::testing::Test {
 protected:
  ParticleMap map;
};

TEST_F(ParticleMapTest, initialization) {
  // Separate maps to mirror original test
  ParticleMap m;
  InsertAll(m, 1, 0.0, 0.0, 1.0);
  InsertAll(m, 2, 0.2, 0.1, 1.0);
  InsertAll(m, 3, 0.3, 0.2, 1.0);

  ParticleMap m1;
  InsertAll(m1, 1, 0.0, 0.0, 1.0);
  InsertAll(m1, 2, 0.2, 0.1, 1.0);
  InsertAll(m1, 3, 0.3, 0.1, 1.0);  // same y as id=2 — ByY conflict
  EXPECT_EQ(m.size(), 3u);
  EXPECT_EQ(m1.size(), 2u);
}

TEST_F(ParticleMapTest, createIncreasessize) {
  auto it = InsertAll(map, 1, 0.0, 0.0, 1.0);
  ASSERT_NE(it, map.cend());
  EXPECT_EQ(map.size(), 1u);
}

TEST_F(ParticleMapTest, createMultiple) {
  InsertAll(map, 1, 0.0, 0.0, 1.0);
  InsertAll(map, 2, 1.0, 1.0, 2.0);
  InsertAll(map, 3, 2.0, 2.0, 1.0);
  EXPECT_EQ(map.size(), 3u);
}

TEST_F(ParticleMapTest, DuplicateIdRejected) {
  InsertAll(map, 1, 0.0, 0.0, 1.0);
  auto it = InsertAll(map, 1, 0.0, 9.0, 9.0);
  EXPECT_EQ(it, map.cend());
  EXPECT_EQ(map.size(), 1u);
}

TEST_F(ParticleMapTest, FindByIdSucceeds) {
  InsertAll(map, 42, 3.0, 4.0, 5.0);
  auto it = map.find_primary(42u);
  ASSERT_NE(it, map.cend());
  EXPECT_EQ(it->id, 42u);
  EXPECT_DOUBLE_EQ(it->x, 3.0);
}

TEST_F(ParticleMapTest, FindMissingIdReturnsEnd) {
  InsertAll(map, 1, 0.0, 0.0, 1.0);
  auto it = map.find_primary(99u);
  EXPECT_EQ(it, map.cend());
}

TEST_F(ParticleMapTest, RemoveBySlot) {
  auto p = InsertAll(map, 1, 0.0, 0.0, 1.0);
  ASSERT_NE(p, map.cend());
  map.remove(*p);
  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.find_primary(1u), map.cend());
}

TEST_F(ParticleMapTest, RemoveByXKey) {
  InsertAll(map, 1, 3.14, 0.0, 1.0);
  bool removed = map.remove<ByX>(3.14);
  EXPECT_TRUE(removed);
  EXPECT_EQ(map.size(), 0u);
}

TEST_F(ParticleMapTest, RemoveByMissingXKey) {
  InsertAll(map, 1, 1.0, 0.0, 1.0);
  bool removed = map.remove<ByX>(99.0);
  EXPECT_FALSE(removed);
  EXPECT_EQ(map.size(), 1u);
}

TEST_F(ParticleMapTest, XIndexIsSorted) {
  InsertAll(map, 1, 3.0, 0.0, 1.0);
  InsertAll(map, 2, 1.0, 0.0, 2.0);
  InsertAll(map, 3, 2.0, 0.0, 3.0);

  auto &xi = map.get<ByX>();
  double prev = -1e9;
  for (auto &p : xi) {
    EXPECT_GE(p.x, prev);
    prev = p.x;
  }
}

TEST_F(ParticleMapTest, MassEqualRange) {
  InsertAll(map, 1, 0.0, 0.0, 5.0);
  InsertAll(map, 2, 1.0, 1.0, 5.0);
  InsertAll(map, 3, 2.0, 2.0, 9.0);

  auto &mi = map.get<ByM>();
  auto [beg, end] = mi.equal_range(5.0);
  std::size_t count = std::distance(beg, end);
  EXPECT_EQ(count, 2u);
  for (auto it = beg; it != end; ++it) EXPECT_DOUBLE_EQ(it->m, 5.0);
}

TEST_F(ParticleMapTest, DistinctMassLevels) {
  InsertAll(map, 1, 0.0, 0.0, 1.0);
  InsertAll(map, 2, 1.0, 1.0, 1.0);
  InsertAll(map, 3, 2.0, 2.0, 2.0);
  InsertAll(map, 4, 3.0, 3.0, 3.0);

  auto &mi = map.get<ByM>();
  std::vector<double> levels;
  for (auto it = mi.begin(); it != mi.cend(); it = mi.upper_bound(it->m))
    levels.push_back(it->m);

  ASSERT_EQ(levels.size(), 3u);
  EXPECT_DOUBLE_EQ(levels[0], 1.0);
  EXPECT_DOUBLE_EQ(levels[1], 2.0);
  EXPECT_DOUBLE_EQ(levels[2], 3.0);
}

TEST_F(ParticleMapTest, ListIndexInsertionOrder) {
  InsertAll(map, 10, 0.0, 0.0, 1.0);
  InsertAll(map, 20, 1.0, 1.0, 2.0);
  InsertAll(map, 30, 2.0, 2.0, 3.0);

  std::vector<uint64_t> ids;
  for (auto &p : map.get<BySeq>()) ids.push_back(p.id);

  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], 10u);
  EXPECT_EQ(ids[1], 20u);
  EXPECT_EQ(ids[2], 30u);
}

TEST_F(ParticleMapTest, PartialIndexing) {
  auto p = map.create(99, 7.0, 8.0, 3.0);
  ASSERT_NE(p, map.cend());

  EXPECT_EQ(map.get<ByX>().size(), 0u);

  map.insert<ByX>(*p);
  EXPECT_EQ(map.get<ByX>().size(), 1u);

  EXPECT_EQ(map.get<BySeq>().size(), 0u);
}

TEST_F(ParticleMapTest, DeindexSecondaryKeepsObject) {
  auto p = InsertAll(map, 1, 1.0, 1.0, 1.0);
  ASSERT_NE(p, map.cend());

  EXPECT_EQ(map.get<ByX>().size(), 1u);
  map.unindex<ByX>(*p);
  EXPECT_EQ(map.get<ByX>().size(), 0u);

  EXPECT_EQ(map.size(), 1u);
  EXPECT_NE(map.find_primary(1u), map.cend());
}

TEST_F(ParticleMapTest, ProjectFromMassToList) {
  auto p = InsertAll(map, 55, 0.0, 0.0, 7.0);
  ASSERT_NE(p, map.cend());

  auto &mi = map.get<ByM>();
  auto mit = mi.find(7.0);
  ASSERT_NE(mit, mi.cend());

  auto lit = map.project<BySeq>(*mit);
  ASSERT_NE(lit, map.get<BySeq>().cend());
  EXPECT_EQ(lit->id, 55u);
}

TEST_F(ParticleMapTest, FullLifecycle) {
  for (uint64_t i = 1; i <= 5; ++i)
    InsertAll(map, i, static_cast<double>(i), 2 * static_cast<double>(i), 1.0);

  EXPECT_EQ(map.size(), 5u);

  map.remove<ByX>(2.0);
  EXPECT_EQ(map.size(), 4u);
  EXPECT_EQ(map.find_primary(2u), map.cend());

  auto it4 = map.find_primary(4u);
  ASSERT_NE(it4, map.cend());
  map.remove(*it4);
  EXPECT_EQ(map.size(), 3u);

  auto &xi = map.get<ByX>();
  std::vector<double> xs;
  for (auto &p : xi) xs.push_back(p.x);
  ASSERT_EQ(xs.size(), 3u);
  EXPECT_DOUBLE_EQ(xs[0], 1.0);
  EXPECT_DOUBLE_EQ(xs[1], 3.0);
  EXPECT_DOUBLE_EQ(xs[2], 5.0);
}

TEST_F(ParticleMapTest, PoolExhaustionReturnsEnd) {
  for (uint64_t i = 0; i < kMaxParticles; ++i) {
    auto p = InsertAll(map, i, static_cast<double>(i),
                        0.5 * static_cast<double>(i), 1.0);
    ASSERT_NE(p, map.cend()) << "Failed at i=" << i;
  }
  EXPECT_EQ(map.size(), kMaxParticles);

  auto p = InsertAll(map, kMaxParticles, 0.0, 0.0, 1.0);
  EXPECT_EQ(p, map.cend());
  EXPECT_EQ(map.size(), kMaxParticles);
}

TEST_F(ParticleMapTest, ReindexXSuccess) {
  auto p = InsertAll(map, 1, 1.0, 3.0, 2.0);
  ASSERT_NE(p, map.cend());

  bool ok = map.modify<optindex::ReindexOnly<ByX>>(*p, [](Particle &pt) { pt.x = 9.0; });
  EXPECT_TRUE(ok);

  EXPECT_NE(map.get<ByX>().find(9.0), map.get<ByX>().cend());
  EXPECT_EQ(map.get<ByX>().find(1.0), map.get<ByX>().cend());
  EXPECT_DOUBLE_EQ(p->x, 9.0);

  EXPECT_NE(map.get<ByY>().find(3.0), map.get<ByY>().cend());
  EXPECT_EQ(map.get<ByM>().size(), 1u);
  EXPECT_EQ(map.get<BySeq>().size(), 1u);
  EXPECT_NE(map.find_primary(1u), map.cend());
}

TEST_F(ParticleMapTest, ReindexXConflictRollback) {
  auto p1 = InsertAll(map, 1, 1.0, 1.0, 1.0);
  auto p2 = InsertAll(map, 2, 5.0, 2.0, 2.0);
  ASSERT_NE(p1, map.cend()); ASSERT_NE(p2, map.cend());

  bool ok = map.modify<optindex::ReindexOnly<ByX>>(*p1, [](Particle &p) { p.x = 5.0; });
  EXPECT_FALSE(ok);

  EXPECT_NE(map.get<ByX>().find(1.0), map.get<ByX>().cend());
  EXPECT_DOUBLE_EQ(p1->x, 1.0);
  EXPECT_NE(map.get<ByX>().find(5.0), map.get<ByX>().cend());
  EXPECT_DOUBLE_EQ(p2->x, 5.0);
  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map.get<ByX>().size(), 2u);

  EXPECT_NE(map.get<ByY>().find(1.0), map.get<ByY>().cend());
  EXPECT_EQ(map.get<ByM>().size(), 2u);
  EXPECT_EQ(map.get<BySeq>().size(), 2u);
}

TEST_F(ParticleMapTest, ReindexMassSuccess) {
  auto p1 = InsertAll(map, 1, 0.0, 0.0, 5.0);
  auto p2 = InsertAll(map, 2, 1.0, 1.0, 5.0);
  ASSERT_NE(p1, map.cend()); ASSERT_NE(p2, map.cend());

  bool ok = map.modify<optindex::ReindexOnly<ByM>>(*p1, [](Particle &p) { p.m = 7.0; });
  EXPECT_TRUE(ok);

  auto &mi = map.get<ByM>();
  auto [b5, e5] = mi.equal_range(5.0);
  EXPECT_EQ(std::distance(b5, e5), 1);
  EXPECT_EQ(b5->id, 2u);

  auto [b7, e7] = mi.equal_range(7.0);
  EXPECT_EQ(std::distance(b7, e7), 1);
  EXPECT_EQ(b7->id, 1u);

  EXPECT_DOUBLE_EQ(p1->m, 7.0);
}

TEST_F(ParticleMapTest, ReindexMassPreservesOtherLevel) {
  InsertAll(map, 1, 0.0, 0.0, 3.0);
  InsertAll(map, 2, 1.0, 1.0, 3.0);
  InsertAll(map, 3, 2.0, 2.0, 3.0);

  auto it1 = map.find_primary(1u);
  ASSERT_NE(it1, map.cend());

  map.modify<optindex::ReindexOnly<ByM>>(*it1, [](Particle &p) { p.m = 9.0; });

  auto &mi = map.get<ByM>();
  auto [b3, e3] = mi.equal_range(3.0);
  EXPECT_EQ(std::distance(b3, e3), 2);

  auto [b9, e9] = mi.equal_range(9.0);
  EXPECT_EQ(std::distance(b9, e9), 1);
}

TEST_F(ParticleMapTest, ReindexExplicitRollbackSuccess) {
  auto p = InsertAll(map, 1, 1.0, 1.0, 1.0);
  ASSERT_NE(p, map.cend());

  bool rollback_called = false;
  bool ok = map.modify<optindex::ReindexOnly<ByX>>(*p, [](Particle &pt) { pt.x = 8.0; });
  EXPECT_TRUE(ok);
  EXPECT_FALSE(rollback_called);
  EXPECT_DOUBLE_EQ(p->x, 8.0);
  EXPECT_NE(map.get<ByX>().find(8.0), map.get<ByX>().cend());
}

TEST_F(ParticleMapTest, ReindexExplicitRollbackConflict) {
  auto p1 = InsertAll(map, 1, 1.0, 1.0, 1.0);
  auto p2 = InsertAll(map, 2, 5.0, 2.0, 2.0);
  ASSERT_NE(p1, map.cend()); ASSERT_NE(p2, map.cend());

  bool ok = map.modify<optindex::ReindexOnly<ByX>>(*p1, [](Particle &p) { p.x = 5.0; });
  EXPECT_FALSE(ok);
  EXPECT_TRUE(p1->x == 1.0);

  EXPECT_DOUBLE_EQ(p1->x, 1.0);
  EXPECT_NE(map.get<ByX>().find(1.0), map.get<ByX>().cend());
  EXPECT_NE(map.get<ByX>().find(5.0), map.get<ByX>().cend());
  EXPECT_EQ(map.get<ByX>().size(), 2u);
  EXPECT_NE(map.get<ByY>().find(1.0), map.get<ByY>().cend());
  EXPECT_EQ(map.size(), 2u);
}

TEST_F(ParticleMapTest, ReindexThenRemove) {
  auto p = InsertAll(map, 1, 1.0, 1.0, 1.0);
  ASSERT_NE(p, map.cend());

  map.modify<optindex::ReindexOnly<ByX>>(*p, [](Particle &pt) { pt.x = 7.0; });
  map.remove(*p);

  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.get<ByX>().size(), 0u);
  EXPECT_EQ(map.get<ByY>().size(), 0u);
  EXPECT_EQ(map.get<ByM>().size(), 0u);
  EXPECT_EQ(map.get<BySeq>().size(), 0u);
}

TEST_F(ParticleMapTest, ReindexYDoesNotDisruptXOrder) {
  InsertAll(map, 1, 1.0, 10.0, 1.0);
  InsertAll(map, 2, 2.0, 20.0, 1.0);
  InsertAll(map, 3, 3.0, 30.0, 1.0);

  auto it2 = map.find_primary(2u);
  ASSERT_NE(it2, map.cend());

  map.modify<optindex::ReindexOnly<ByY>>(*it2, [](Particle &p) { p.y = 99.0; });

  std::vector<double> xs;
  for (auto &p : map.get<ByX>()) xs.push_back(p.x);
  ASSERT_EQ(xs.size(), 3u);
  EXPECT_DOUBLE_EQ(xs[0], 1.0);
  EXPECT_DOUBLE_EQ(xs[1], 2.0);
  EXPECT_DOUBLE_EQ(xs[2], 3.0);

  EXPECT_EQ(map.get<ByY>().find(20.0), map.get<ByY>().cend());
  EXPECT_NE(map.get<ByY>().find(99.0), map.get<ByY>().cend());
}

TEST_F(ParticleMapTest, createAllIndicesSecondaryConflictFullRollback) {
  auto p1 = InsertAll(map, 1, 5.0, 1.0, 1.0);
  ASSERT_NE(p1, map.cend());
  EXPECT_EQ(map.size(), 1u);

  // Particle 2 with same x=5.0 — ByX insert will fail
  auto p2 = InsertAll(map, 2, 5.0, 2.0, 2.0);
  EXPECT_EQ(p2, map.cend());

  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map.find_primary(2u), map.cend());
  EXPECT_EQ(map.get<ByX>().size(), 1u);
  EXPECT_EQ(map.get<ByY>().size(), 1u);
  EXPECT_EQ(map.get<ByM>().size(), 1u);
  EXPECT_EQ(map.get<BySeq>().size(), 1u);

  EXPECT_NE(map.find_primary(1u), map.cend());
  EXPECT_DOUBLE_EQ(p1->x, 5.0);
}

TEST_F(ParticleMapTest, ProjectNotInTargetIndexReturnsEnd) {
  auto p = map.create(1, 1.0, 1.0, 1.0);
  ASSERT_NE(p, map.cend());

  EXPECT_EQ(map.project<ByX>(*p),   map.get<ByX>().cend());
  EXPECT_EQ(map.project<ByY>(*p),   map.get<ByY>().cend());
  EXPECT_EQ(map.project<ByM>(*p),   map.get<ByM>().cend());
  EXPECT_EQ(map.project<BySeq>(*p), map.get<BySeq>().cend());

  map.insert<ByX>(*p);
  EXPECT_NE(map.project<ByX>(*p),   map.get<ByX>().cend());
  EXPECT_EQ(map.project<ByY>(*p),   map.get<ByY>().cend());
}

TEST_F(ParticleMapTest, ReindexTwiceSameSlot) {
  auto p = InsertAll(map, 1, 1.0, 1.0, 1.0);
  ASSERT_NE(p, map.cend());

  bool ok1 = map.modify<optindex::ReindexOnly<ByX>>(*p, [](Particle &pt) { pt.x = 5.0; });
  EXPECT_TRUE(ok1);
  EXPECT_DOUBLE_EQ(p->x, 5.0);
  EXPECT_NE(map.get<ByX>().find(5.0), map.get<ByX>().cend());
  EXPECT_EQ(map.get<ByX>().find(1.0), map.get<ByX>().cend());

  bool ok2 = map.modify<optindex::ReindexOnly<ByX>>(*p, [](Particle &pt) { pt.x = 9.0; });
  EXPECT_TRUE(ok2);
  EXPECT_DOUBLE_EQ(p->x, 9.0);
  EXPECT_NE(map.get<ByX>().find(9.0), map.get<ByX>().cend());
  EXPECT_EQ(map.get<ByX>().find(5.0), map.get<ByX>().cend());

  EXPECT_EQ(map.size(), 1u);
  EXPECT_EQ(map.get<ByX>().size(), 1u);
  EXPECT_NE(map.get<ByY>().find(1.0), map.get<ByY>().cend());
  EXPECT_EQ(map.get<ByM>().size(), 1u);
  EXPECT_EQ(map.get<BySeq>().size(), 1u);
  EXPECT_NE(map.find_primary(1u), map.cend());

  bool ok3 = map.modify<optindex::ReindexOnly<ByX>>(*p, [](Particle &pt) { pt.x = 1.0; });
  EXPECT_TRUE(ok3);
  EXPECT_DOUBLE_EQ(p->x, 1.0);
  EXPECT_NE(map.get<ByX>().find(1.0), map.get<ByX>().cend());
}

// KeyFrom variant
using ParticleMapKF = optindex::FixedSizeOptIndex<
    Particle, kMaxParticles,
    optindex::Unordered<optindex::KeyFrom<&Particle::id>, IdHash, IdEqual, 128>,
    optindex::IndexBy<optindex::Ordered<optindex::KeyFrom<&Particle::x>, std::less<double>>, ByX>,
    optindex::IndexBy<optindex::Ordered<optindex::KeyFrom<&Particle::y>, std::less<double>>, ByY>,
    optindex::IndexBy<optindex::OrderedNonUnique<optindex::KeyFrom<&Particle::m>, std::less<double>>, ByM>,
    optindex::IndexBy<optindex::List, BySeq>
>;

class KeyFromTest : public ::testing::Test {
 protected:
  ParticleMapKF map;
};

TEST_F(KeyFromTest, KeyFromDataMemberFind) {
  auto p = map.create_all(42, 1.0, 2.0, 3.0);
  ASSERT_NE(p, map.cend());
  auto found = map.find_primary(42u);
  ASSERT_NE(found, map.cend());
  EXPECT_EQ(found->id, 42u);
}

TEST_F(KeyFromTest, KeyFromDataMemberOrdering) {
  map.create_all(1, 3.0, 0.0, 1.0);
  map.create_all(2, 1.0, 0.0, 2.0);
  map.create_all(3, 2.0, 0.0, 3.0);

  double prev = -1e9;
  for (auto &p : map.get<ByX>()) {
    EXPECT_GE(p.x, prev);
    prev = p.x;
  }
}

TEST_F(KeyFromTest, KeyFromDataMemberNonUnique) {
  map.create_all(1, 0.0, 0.0, 7.0);
  map.create_all(2, 1.0, 1.0, 7.0);
  map.create_all(3, 2.0, 2.0, 9.0);

  auto [beg, end] = map.get<ByM>().equal_range(7.0);
  EXPECT_EQ(std::distance(beg, end), 2);
}

TEST_F(KeyFromTest, KeyFromDataMemberDuplicateRejected) {
  map.create_all(1, 0.0, 0.0, 1.0);
  auto p = map.create_all(1, 9.0, 9.0, 9.0);
  EXPECT_EQ(p, map.cend());
  EXPECT_EQ(map.size(), 1u);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
