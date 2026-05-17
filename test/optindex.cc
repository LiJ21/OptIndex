#include "optindex.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal object types used across tests
// ---------------------------------------------------------------------------

struct Obj {
  int id;
  double val;
  double key;
};

struct TagA {};
struct TagB {};
struct TagC {};
struct TagD {};

// ---------------------------------------------------------------------------
// Layout tests: kWidths, kOffsets, kMasks, StorageType, kTotalWidth
// ---------------------------------------------------------------------------

TEST(BitfieldState, EmptyHasZeroWidth) {
  using L = optindex::detail::BitfieldState<>;
  static_assert(L::kNumSlots == 0);
  static_assert(L::kTotalWidth == 0);
}

TEST(BitfieldState, SingleK1HasWidth1) {
  // K=1 → bit_width(1) == 1
  using IB = optindex::IndexBy<optindex::List, TagA>;
  using L = optindex::detail::BitfieldState<IB>;
  static_assert(L::kNumSlots == 1);
  static_assert(L::kWidths[0] == 1);
  static_assert(L::kOffsets[0] == 0);
  static_assert(L::kMasks[0] == 1);
  static_assert(L::kTotalWidth == 1);
  SUCCEED();
}

TEST(BitfieldState, K2HasWidth2) {
  // K=2 → bit_width(2) == 2 (values 0,1,2 need 2 bits)
  using IB = optindex::IndexBy<optindex::List, TagA, TagB>;
  using L = optindex::detail::BitfieldState<IB>;
  static_assert(L::kWidths[0] == 2);
  static_assert(L::kMasks[0] == 0b11);
  SUCCEED();
}

TEST(BitfieldState, K3HasWidth2) {
  // K=3 → bit_width(3) == 2 (values 0..3 fit in 2 bits)
  using IB = optindex::IndexBy<optindex::List, TagA, TagB, TagC>;
  using L = optindex::detail::BitfieldState<IB>;
  static_assert(L::kWidths[0] == 2);
  SUCCEED();
}

TEST(BitfieldState, K4HasWidth3) {
  // K=4 → bit_width(4) == 3 (values 0..4 need 3 bits)
  using IB = optindex::IndexBy<optindex::List, TagA, TagB, TagC, TagD>;
  using L = optindex::detail::BitfieldState<IB>;
  static_assert(L::kWidths[0] == 3);
  SUCCEED();
}

TEST(BitfieldState, TwoSlotsOffsets) {
  using IB1 = optindex::IndexBy<optindex::List, TagA>;        // width=1, offset=0
  using IB2 = optindex::IndexBy<optindex::List, TagB, TagC>;  // width=2, offset=1
  using L = optindex::detail::BitfieldState<IB1, IB2>;
  static_assert(L::kNumSlots == 2);
  static_assert(L::kOffsets[0] == 0);
  static_assert(L::kOffsets[1] == 1);
  static_assert(L::kTotalWidth == 3);
  SUCCEED();
}

TEST(BitfieldState, StorageUint8For8Bits) {
  // 8 slots each K=1 -> kTotalWidth=8 -> StorageType=uint8_t
  using IB = optindex::IndexBy<optindex::List, TagA>;
  using L = optindex::detail::BitfieldState<IB, IB, IB, IB, IB, IB, IB, IB>;
  static_assert(L::kTotalWidth == 8);
  static_assert(std::is_same_v<L::StorageType, uint8_t>);
  SUCCEED();
}

TEST(BitfieldState, StorageUint16For9Bits) {
  using IB1 = optindex::IndexBy<optindex::List, TagA>;  // 1 bit each
  using L = optindex::detail::BitfieldState<IB1, IB1, IB1, IB1, IB1, IB1, IB1, IB1, IB1>;
  static_assert(L::kTotalWidth == 9);
  static_assert(std::is_same_v<L::StorageType, uint16_t>);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// BitfieldState round-trip tests
// ---------------------------------------------------------------------------

TEST(BitfieldState, SingleSlotRoundTrip) {
  using IB = optindex::IndexBy<optindex::List, TagA>;
  optindex::detail::BitfieldState<IB> bf;

  EXPECT_EQ(bf.get_slot<0>(), 0u);
  bf.set_slot<0>(1);
  EXPECT_EQ(bf.get_slot<0>(), 1u);
  bf.set_slot<0>(0);
  EXPECT_EQ(bf.get_slot<0>(), 0u);
}

TEST(BitfieldState, TwoSlotsIndependent) {
  using IB1 = optindex::IndexBy<optindex::List, TagA>;
  using IB2 = optindex::IndexBy<optindex::List, TagB, TagC>;  // 2-bit slot
  optindex::detail::BitfieldState<IB1, IB2> bf;

  bf.set_slot<0>(1);
  bf.set_slot<1>(2);
  EXPECT_EQ(bf.get_slot<0>(), 1u);
  EXPECT_EQ(bf.get_slot<1>(), 2u);

  bf.set_slot<0>(0);
  EXPECT_EQ(bf.get_slot<0>(), 0u);
  EXPECT_EQ(bf.get_slot<1>(), 2u);  // untouched

  bf.set_slot<1>(0);
  EXPECT_EQ(bf.get_slot<0>(), 0u);
  EXPECT_EQ(bf.get_slot<1>(), 0u);
}

TEST(BitfieldState, K2AllValues) {
  using IB = optindex::IndexBy<optindex::List, TagA, TagB>;  // values 0,1,2
  optindex::detail::BitfieldState<IB> bf;

  for (size_t v = 0; v <= 2; ++v) {
    bf.set_slot<0>(v);
    EXPECT_EQ(bf.get_slot<0>(), v);
  }
}

TEST(BitfieldState, MaskClampsOverflow) {
  // Writing 3 into a 1-bit slot should be masked to 1.
  using IB = optindex::IndexBy<optindex::List, TagA>;  // 1-bit slot
  optindex::detail::BitfieldState<IB> bf;
  bf.set_slot<0>(3);  // 3 & 1 = 1
  EXPECT_EQ(bf.get_slot<0>(), 1u);
}

// ---------------------------------------------------------------------------
// TagTraits tests
// ---------------------------------------------------------------------------

TEST(TagTraits, SlotIndexOf) {
  using IB1 = optindex::IndexBy<optindex::List, TagA>;
  using IB2 = optindex::IndexBy<optindex::List, TagB>;
  using Traits = optindex::detail::TagTraits<IB1, IB2>;
  static_assert(Traits::slot_index_of<TagA> == 0);
  static_assert(Traits::slot_index_of<TagB> == 1);
  SUCCEED();
}

TEST(TagTraits, TagValueWithinSlot) {
  using IB = optindex::IndexBy<optindex::List, TagA, TagB>;
  using Traits = optindex::detail::TagTraits<IB>;
  static_assert(Traits::tag_value_within_slot<TagA> == 1);
  static_assert(Traits::tag_value_within_slot<TagB> == 2);
  SUCCEED();
}

TEST(TagTraits, ContainerIndexOfMultiSlot) {
  using IB1 = optindex::IndexBy<optindex::List, TagA, TagB>;
  using IB2 = optindex::IndexBy<optindex::List, TagC, TagD>;
  using Traits = optindex::detail::TagTraits<IB1, IB2>;
  // flat AllTagsTuple = <TagA, TagB, TagC, TagD>
  static_assert(Traits::container_index_of<TagA> == 0);
  static_assert(Traits::container_index_of<TagB> == 1);
  static_assert(Traits::container_index_of<TagC> == 2);
  static_assert(Traits::container_index_of<TagD> == 3);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// FixedSizeOptIndex: K=1 basic operations
// ---------------------------------------------------------------------------

namespace {
struct ObjId { int id; };

using PrimaryIdx = optindex::Ordered<optindex::KeyFrom<&ObjId::id>, std::less<int>>;
using Map1 = optindex::FixedSizeOptIndex<ObjId, 16, PrimaryIdx,
                                       optindex::IndexBy<optindex::List, TagA>>;
}  // namespace

TEST(FixedSizeOptIndex, CreateAndSize) {
  Map1 m;
  EXPECT_EQ(m.size(), 0u);
  auto p = m.create_all(42);
  ASSERT_NE(p, m.cend());
  EXPECT_EQ(m.size(), 1u);
  EXPECT_EQ(p->id, 42);
}

TEST(FixedSizeOptIndex, FindPrimary) {
  Map1 m;
  m.create_all(10);
  m.create_all(20);
  auto it = m.find_primary(10);
  ASSERT_NE(it, m.cend());
  EXPECT_EQ(it->id, 10);
  EXPECT_EQ(m.find_primary(99), m.cend());
}

TEST(FixedSizeOptIndex, RemoveBySlot) {
  Map1 m;
  auto p = m.create_all(5);
  ASSERT_NE(p, m.cend());
  m.remove(*p);
  EXPECT_EQ(m.size(), 0u);
  EXPECT_EQ(m.find_primary(5), m.cend());
}

TEST(FixedSizeOptIndex, DeindexSecondary) {
  Map1 m;
  auto p = m.create_all(7);
  ASSERT_NE(p, m.cend());
  EXPECT_EQ(m.get<TagA>().size(), 1u);
  bool removed = m.unindex<TagA>(*p);
  EXPECT_TRUE(removed);
  EXPECT_EQ(m.get<TagA>().size(), 0u);
  EXPECT_EQ(m.size(), 1u);  // still in primary

  bool removed2 = m.unindex<TagA>(*p);
  EXPECT_FALSE(removed2);  // idempotent
}

TEST(FixedSizeOptIndex, ReinsertAfterDeindex) {
  Map1 m;
  auto p = m.create_all(3);
  m.unindex<TagA>(*p);
  EXPECT_EQ(m.get<TagA>().size(), 0u);
  m.insert<TagA>(*p);
  EXPECT_EQ(m.get<TagA>().size(), 1u);
}

TEST(FixedSizeOptIndex, PoolFull) {
  Map1 m;
  for (int i = 0; i < 16; ++i)
    ASSERT_NE(m.create_all(i), m.cend());
  EXPECT_EQ(m.create_all(99), m.cend());  // pool exhausted
  EXPECT_EQ(m.size(), 16u);
}

// ---------------------------------------------------------------------------
// FixedSizeOptIndex: K>1 multi-tag slot
// ---------------------------------------------------------------------------

namespace {
struct SimpleObj { int id; double val; };

using PrimaryK2 = optindex::Ordered<optindex::KeyFrom<&SimpleObj::id>, std::less<int>>;
// One slot with two tags sharing the same List hook
using MapK2 = optindex::FixedSizeOptIndex<SimpleObj, 32, PrimaryK2,
    optindex::IndexBy<optindex::List, TagA, TagB>>;
}  // namespace

TEST(FixedSizeOptIndexK2, CreateInTagA) {
  MapK2 m;
  auto p = m.create(1, 1.0);
  ASSERT_NE(p, m.cend());
  m.insert<TagA>(*p);
  EXPECT_EQ(m.get<TagA>().size(), 1u);
  EXPECT_EQ(m.get<TagB>().size(), 0u);
}

TEST(FixedSizeOptIndexK2, CreateInTagB) {
  MapK2 m;
  auto p = m.create(2, 2.0);
  ASSERT_NE(p, m.cend());
  m.insert<TagB>(*p);
  EXPECT_EQ(m.get<TagA>().size(), 0u);
  EXPECT_EQ(m.get<TagB>().size(), 1u);
}

TEST(FixedSizeOptIndexK2, MigrateAtoB) {
  MapK2 m;
  auto p = m.create(3, 3.0);
  EXPECT_TRUE(m.insert<TagA>(*p));
  ASSERT_EQ(m.get<TagA>().size(), 1u);

  // insert while already linked returns false
  EXPECT_FALSE(m.insert<TagB>(*p));
  EXPECT_EQ(m.get<TagA>().size(), 1u);

  // explicit deindex then insert
  EXPECT_TRUE(m.unindex<TagA>(*p));
  EXPECT_TRUE(m.insert<TagB>(*p));
  EXPECT_EQ(m.get<TagA>().size(), 0u);
  EXPECT_EQ(m.get<TagB>().size(), 1u);
}

TEST(FixedSizeOptIndexK2, MigrateBtoA) {
  MapK2 m;
  auto p = m.create(4, 4.0);
  EXPECT_TRUE(m.insert<TagB>(*p));
  EXPECT_FALSE(m.insert<TagA>(*p));  // slot occupied
  EXPECT_TRUE(m.unindex<TagB>(*p));
  EXPECT_TRUE(m.insert<TagA>(*p));
  EXPECT_EQ(m.get<TagA>().size(), 1u);
  EXPECT_EQ(m.get<TagB>().size(), 0u);
}

TEST(FixedSizeOptIndexK2, DeindexTagB) {
  MapK2 m;
  auto p = m.create(5, 5.0);
  m.insert<TagB>(*p);
  EXPECT_TRUE(m.unindex<TagB>(*p));
  EXPECT_EQ(m.get<TagB>().size(), 0u);
  EXPECT_FALSE(m.unindex<TagB>(*p));  // already absent
}

TEST(FixedSizeOptIndexK2, DeindexWrongTagReturnsFalse) {
  MapK2 m;
  auto p = m.create(6, 6.0);
  m.insert<TagA>(*p);
  // Not in TagB → false
  EXPECT_FALSE(m.unindex<TagB>(*p));
  EXPECT_EQ(m.get<TagA>().size(), 1u);  // TagA untouched
}

// ---------------------------------------------------------------------------
// FixedSizeOptIndex: modify<Tag> reindex
// ---------------------------------------------------------------------------

namespace {
struct ValObj { int id; double val; };

using PrimaryMod = optindex::Ordered<optindex::KeyFrom<&ValObj::id>, std::less<int>>;
using SecByVal = optindex::IndexBy<
    optindex::OrderedNonUnique<optindex::KeyFrom<&ValObj::val>, std::less<double>>,
    TagA>;
using MapMod = optindex::FixedSizeOptIndex<ValObj, 32, PrimaryMod, SecByVal>;
}  // namespace

TEST(FixedSizeOptIndexModify, ReindexAfterMutation) {
  MapMod m;
  auto p1 = m.create_all(1, 10.0);
  auto p2 = m.create_all(2, 20.0);
  (void)p2;

  // Items in val order: p1(10), p2(20)
  auto &idx = m.get<TagA>();
  auto it = idx.begin();
  EXPECT_DOUBLE_EQ(it->val, 10.0);

  bool ok = m.modify<optindex::ReindexOnly<TagA>>(*p1, [](ValObj &o) { o.val = 30.0; });
  EXPECT_TRUE(ok);

  // Now order should be p2(20), p1(30)
  it = idx.begin();
  EXPECT_DOUBLE_EQ(it->val, 20.0);
  ++it;
  EXPECT_DOUBLE_EQ(it->val, 30.0);
}

TEST(FixedSizeOptIndexModify, ModifyNotLinked) {
  MapMod m;
  auto p = m.create(1, 5.0);  // in primary only, NOT in secondary
  // modify should succeed trivially (was_linked=false, nothing to reindex)
  bool ok = m.modify<optindex::ReindexOnly<TagA>>(*p, [](ValObj &o) { o.val = 99.0; });
  EXPECT_TRUE(ok);
  EXPECT_DOUBLE_EQ(p->val, 99.0);
  EXPECT_EQ(m.get<TagA>().size(), 0u);
}

// ---------------------------------------------------------------------------
// FixedSizeOptIndex: remove<Tag>(key)
// ---------------------------------------------------------------------------

TEST(FixedSizeOptIndex, RemoveByKey) {
  MapMod m;
  m.create_all(10, 1.0);
  m.create_all(20, 2.0);
  EXPECT_EQ(m.size(), 2u);
  bool removed = m.remove<TagA>(1.0);
  EXPECT_TRUE(removed);
  EXPECT_EQ(m.size(), 1u);
  EXPECT_FALSE(m.remove<TagA>(99.0));
}

// ---------------------------------------------------------------------------
// Multiple secondary slots (two independent IndexBy declarations)
// ---------------------------------------------------------------------------

namespace {
struct MultiObj { int id; int cat; double score; };

struct ByCat {};
struct ByScore {};

using PrimaryMulti = optindex::Ordered<optindex::KeyFrom<&MultiObj::id>, std::less<int>>;
using MapMulti = optindex::FixedSizeOptIndex<MultiObj, 64, PrimaryMulti,
    optindex::IndexBy<optindex::List, ByCat>,
    optindex::IndexBy<optindex::OrderedNonUnique<
        optindex::KeyFrom<&MultiObj::score>, std::less<double>>, ByScore>>;
}  // namespace

TEST(FixedSizeOptIndexMultiSlot, IndependentSlots) {
  MapMulti m;
  auto p = m.create_all(1, 0, 3.14);
  ASSERT_NE(p, m.cend());
  EXPECT_EQ(m.get<ByCat>().size(), 1u);
  EXPECT_EQ(m.get<ByScore>().size(), 1u);

  m.unindex<ByCat>(*p);
  EXPECT_EQ(m.get<ByCat>().size(), 0u);
  EXPECT_EQ(m.get<ByScore>().size(), 1u);  // untouched
}

TEST(FixedSizeOptIndexMultiSlot, RemoveUnlinksAllSlots) {
  MapMulti m;
  auto p = m.create_all(2, 1, 2.72);
  EXPECT_EQ(m.size(), 1u);
  m.remove(*p);
  EXPECT_EQ(m.size(), 0u);
  EXPECT_EQ(m.get<ByCat>().size(), 0u);
  EXPECT_EQ(m.get<ByScore>().size(), 0u);
}

TEST(FixedSizeOptIndexMultiSlot, ModifyBothTags) {
  MapMulti m;
  m.create_all(1, 0, 1.0);
  m.create_all(2, 0, 2.0);
  auto it = m.find_primary(1);
  ASSERT_NE(it, m.cend());
  bool ok = m.modify<optindex::ReindexOnly<ByScore>>(*it, [](MultiObj &o) { o.score = 3.0; });
  EXPECT_TRUE(ok);
  EXPECT_DOUBLE_EQ(it->score, 3.0);
}

// ---------------------------------------------------------------------------
// List as primary index (no key, insertion order preserved)
// ---------------------------------------------------------------------------

namespace {
struct ListPrimaryOrderedTag {};
using MapListPrimary = optindex::FixedSizeOptIndex<
    ValObj, 16, optindex::List,
    optindex::IndexBy<optindex::OrderedNonUnique<
                          optindex::KeyFrom<&ValObj::val>, std::less<double>>,
                      ListPrimaryOrderedTag>>;
}  // namespace

TEST(FixedSizeOptIndexListPrimary, InsertIterateRemove) {
  MapListPrimary m;
  auto a = m.create_all(1, 10.0);
  auto b = m.create_all(2, 20.0);
  auto c = m.create_all(3, 30.0);
  ASSERT_NE(a, m.cend()); ASSERT_NE(b, m.cend()); ASSERT_NE(c, m.cend());

  EXPECT_EQ(m.size(), 3u);

  // Primary iteration preserves insertion order.
  std::vector<int> ids;
  for (auto &o : m) ids.push_back(o.id);
  EXPECT_EQ(ids, (std::vector<int>{1, 2, 3}));

  // Secondary still works.
  EXPECT_EQ(m.get<ListPrimaryOrderedTag>().size(), 3u);
  EXPECT_NE(m.get<ListPrimaryOrderedTag>().find(20.0), m.get<ListPrimaryOrderedTag>().cend());

  // remove unlinks from both primary and secondary.
  m.remove(*b);
  EXPECT_EQ(m.size(), 2u);
  EXPECT_EQ(m.get<ListPrimaryOrderedTag>().find(20.0), m.get<ListPrimaryOrderedTag>().cend());

  ids.clear();
  for (auto &o : m) ids.push_back(o.id);
  EXPECT_EQ(ids, (std::vector<int>{1, 3}));
}
