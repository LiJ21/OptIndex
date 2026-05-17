#ifndef OPTINDEX_H
#define OPTINDEX_H

#include <array>
#include <bit>
#include <bitset>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/intrusive/unordered_set.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace bi = boost::intrusive;

namespace optindex {

#if defined(ALIGNED_WITH_CACHELINE) && ALIGNED_WITH_CACHELINE
constexpr size_t kCacheLineSize = 64;
#endif
// ===============================================================================
// Index and hook helpers
// ===============================================================================

template <auto tMemberPtr> class KeyFrom {
  template <auto MemberPtr> struct member_type;
  template <typename C, typename T, T C::*MemberPtr>
  struct member_type<MemberPtr> {
    using type = T;
  };

public:
  auto operator()(const auto &obj) const {
    return std::invoke(tMemberPtr, obj);
  }
  using type = typename member_type<tMemberPtr>::type;
};

template <typename THook, typename... TTags> struct IndexBy {
  static_assert(sizeof...(TTags) >= 1, "IndexBy requires at least one tag");
  using hook_type = THook;
  using tags_tuple = std::tuple<TTags...>;
  using Tag = std::tuple_element_t<0, tags_tuple>;
};

// Helper to generate tN index tags for 0, ..., tN - 1
template <typename THook, size_t tN> struct IndexByFamily {
  template <size_t tIDX> struct Index {};

private:
  template <typename Seq> struct make_iby;
  template <size_t... Is> struct make_iby<std::index_sequence<Is...>> {
    using type = IndexBy<THook, Index<Is>...>;
  };

public:
  using type = typename make_iby<std::make_index_sequence<tN>>::type;
};

template <typename THook, size_t tN>
using IndexByFamily_t = typename IndexByFamily<THook, tN>::type;

// Pool allocator backed by a fixed-size freelist.
// Satisfies the allocator interface used by std::allocator_traits:
//   value_type, allocate(n), deallocate(ptr, n).
// allocate() returns nullptr when the pool is exhausted instead of throwing
// std::bad_alloc.
template <typename T, size_t N> class FixedSizeLifoPool {
public:
  using value_type = T;
  static constexpr size_t kSize = N;

  FixedSizeLifoPool() {
    mem_pool_.reserve(kSize);
    for (auto i : std::views::iota(size_t{0}, kSize)) {
      MemSlot s;
      s.next_free_idx_ = i + 1;
      mem_pool_.push_back(s);
    }
  }

  T *allocate(size_t /* n */) noexcept {
    if (free_head_ == kSize)
      return nullptr;
    auto idx = free_head_;
    free_head_ = mem_pool_[idx].next_free_idx_;
    --free_count_;
    return reinterpret_cast<T *>(&mem_pool_[idx]);
  }
  void deallocate(T *ptr, size_t /* n */) noexcept {
    auto idx = get_index(*ptr);
    mem_pool_[idx].next_free_idx_ = free_head_;
    free_head_ = idx;
    ++free_count_;
  }

  T *create(auto &&...args) {
    T *p = allocate(1);
    if (!p)
      return nullptr;
    std::construct_at(p, std::forward<decltype(args)>(args)...);
    return p;
  }
  void remove(T &obj) {
    std::destroy_at(&obj);
    deallocate(&obj, 1);
  }

  bool owns(T &obj) const { return get_index(obj) < mem_pool_.size(); }
  size_t size() const { return kSize - free_count_; }
  size_t capacity() const { return kSize; }
  bool empty() const { return kSize == free_count_; }
  bool full() const { return free_count_ == 0; }

  ~FixedSizeLifoPool() {
    std::bitset<kSize> live_mask{};
    size_t idx = free_head_;
    live_mask.set();
    while (idx < kSize) {
      live_mask.reset(idx);
      idx = mem_pool_[idx].next_free_idx_;
    }
    for (size_t i = 0; i < mem_pool_.size(); ++i)
      if (live_mask.test(i))
        std::destroy_at(reinterpret_cast<T *>(&mem_pool_[i]));
  }

private:
  size_t get_index(T &obj) const {
    return static_cast<size_t>(reinterpret_cast<MemSlot *>(&obj) -
                               mem_pool_.data());
  }
#if defined(ALIGNED_WITH_CACHELINE) && ALIGNED_WITH_CACHELINE
  struct alignas(kCacheLineSize) MemSlot {
#else
  struct alignas(T) MemSlot {
#endif
    union {
      std::byte storage[sizeof(T)];
      size_t next_free_idx_;
    };
  };
  std::vector<MemSlot> mem_pool_;
  size_t free_head_{0};
  size_t free_count_{kSize};
};

template <size_t N> struct FixedSizeAllocator {
  template <typename T> using type = FixedSizeLifoPool<T, N>;
};

template <size_t tIDX> struct HookTag {};

template <typename T>
concept COrderedIndex = requires {
  typename T::KeyGetter;
  typename T::Compare;
};

struct List {};
template <typename TKeyGetter, typename TCompare> struct Ordered {
  using KeyGetter = TKeyGetter;
  using Compare = TCompare;
};
template <typename TKeyGetter, typename THash, typename TEqual, size_t tBuckets>
struct Unordered {
  using KeyGetter = TKeyGetter;
  using Hash = THash;
  using Equal = TEqual;
  static constexpr size_t kBuckets{tBuckets};
};
template <typename TKeyGetter, typename TCompare> struct OrderedNonUnique {
  using KeyGetter = TKeyGetter;
  using Compare = TCompare;
};

namespace detail {

// Allocate n objects from different allocator contracts, allowing for std
// compatible exception and a lightweight nullptr for failure
template <typename TAlloc>
[[nodiscard]] typename std::allocator_traits<TAlloc>::pointer
try_allocate(TAlloc &alloc, size_t n) noexcept {
  using Traits = std::allocator_traits<TAlloc>;
  if constexpr (noexcept(alloc.allocate(n))) {
    return Traits::allocate(alloc, n);
  } else {
    try {
      return Traits::allocate(alloc, n);
    } catch (const std::bad_alloc &) {
      return nullptr;
    }
  }
}

template <typename TBucketType, size_t N> struct BucketBase {
  TBucketType buckets_[N];
};

template <typename TIndexType, size_t tIDX> struct HookTrait;
template <size_t tIDX> struct HookTrait<List, tIDX> : List {
  using Hook = bi::list_base_hook<bi::link_mode<bi::normal_link>,
                                  bi::tag<HookTag<tIDX>>>;
  template <typename TObject>
  using Container =
      bi::list<TObject, bi::base_hook<Hook>, bi::constant_time_size<true>>;
  static auto insert(auto &container, auto &obj) {
    container.push_back(obj);
    return container.iterator_to(obj);
  }
};
template <size_t tIDX, typename TKeyGetter, typename TCompare>
struct HookTrait<Ordered<TKeyGetter, TCompare>, tIDX>
    : Ordered<TKeyGetter, TCompare> {
  using Hook =
      bi::set_base_hook<bi::link_mode<bi::normal_link>, bi::tag<HookTag<tIDX>>>;
  template <typename TObject>
  using Container =
      bi::set<TObject, bi::base_hook<Hook>, bi::constant_time_size<true>,
              bi::key_of_value<TKeyGetter>, bi::compare<TCompare>,
              bi::optimize_size<false>>;
  static auto insert(auto &container, auto &obj) {
    typename std::remove_reference_t<decltype(container)>::insert_commit_data c;
    auto [it, ok] = container.insert_check(TKeyGetter{}(obj), c);
    if (!ok)
      return container.end();
    return container.insert_commit(obj, c);
  }
};
template <size_t tIDX, typename TKeyGetter, typename TCompare>
struct HookTrait<OrderedNonUnique<TKeyGetter, TCompare>, tIDX>
    : OrderedNonUnique<TKeyGetter, TCompare> {
  using Hook =
      bi::set_base_hook<bi::link_mode<bi::normal_link>, bi::tag<HookTag<tIDX>>>;
  template <typename TObject>
  using Container =
      bi::multiset<TObject, bi::base_hook<Hook>, bi::constant_time_size<true>,
                   bi::key_of_value<TKeyGetter>, bi::compare<TCompare>>;
  static auto insert(auto &container, auto &obj) {
    return container.insert(obj);
  }
};
template <size_t tIDX, typename TKeyGetter, typename THash, typename TEqual,
          size_t tBuckets>
struct HookTrait<Unordered<TKeyGetter, THash, TEqual, tBuckets>, tIDX>
    : public Unordered<TKeyGetter, THash, TEqual, tBuckets> {
  using Hook = bi::unordered_set_base_hook<bi::link_mode<bi::normal_link>,
                                           bi::tag<HookTag<tIDX>>>;
  template <typename TObject>
  struct Container
      : BucketBase<typename bi::unordered_set<TObject,
                                              bi::base_hook<Hook>>::bucket_type,
                   tBuckets>,
        bi::unordered_set<
            TObject, bi::base_hook<Hook>, bi::key_of_value<TKeyGetter>,
            bi::constant_time_size<true>, bi::hash<THash>, bi::equal<TEqual>> {
    using SetType = bi::unordered_set<
        TObject, bi::base_hook<Hook>, bi::key_of_value<TKeyGetter>,
        bi::constant_time_size<true>, bi::hash<THash>, bi::equal<TEqual>>;
    using BucketType = typename SetType::bucket_type;
    using Base = BucketBase<BucketType, tBuckets>;
    using BucketTraits = typename SetType::bucket_traits;
    Container() : Base{}, SetType(BucketTraits(Base::buckets_, tBuckets)) {}
  };
  static auto insert(auto &container, auto &obj) {
    auto [it, ok] = container.insert(obj);
    if (!ok)
      return container.end();
    return it;
  }
};

template <typename...> inline constexpr bool always_false_v = false;

// ===============================================================================
// Index and hook infra below. We call List, Ordered and so on a hook. An object
// holds hooks and can be inserted into one (or none) of multiple containers
// (indices) for each hook.
// ===============================================================================

// Set per-hook bit width to minimum bits to encode values 0..K (absent -> 0,
// tag_k
// -> k).
template <typename TIndexBy>
inline constexpr size_t kSlotWidth =
    std::bit_width(std::tuple_size_v<typename TIndexBy::tags_tuple>);

template <size_t tBits>
using storage_type_t = std::conditional_t<
    tBits <= 8, uint8_t,
    std::conditional_t<
        tBits <= 16, uint16_t,
        std::conditional_t<tBits <= 32, uint32_t,
                           std::conditional_t<tBits <= 64, uint64_t, void>>>>;

template <typename... tSecondaryIndexBys> struct BitfieldState {
  static constexpr size_t kNumSlots = sizeof...(tSecondaryIndexBys);

  static constexpr std::array<size_t, kNumSlots> kWidths = {
      kSlotWidth<tSecondaryIndexBys>...};

  static constexpr std::array<size_t, kNumSlots> kOffsets = []() {
    std::array<size_t, kNumSlots> arr{};
    size_t acc = 0;
    for (size_t i = 0; i < kNumSlots; ++i) {
      arr[i] = acc;
      acc += kWidths[i];
    }
    return arr;
  }();

  static constexpr std::array<size_t, kNumSlots> kMasks = []() {
    std::array<size_t, kNumSlots> arr{};
    for (size_t i = 0; i < kNumSlots; ++i)
      arr[i] = (size_t{1} << kWidths[i]) - 1;
    return arr;
  }();

  static constexpr size_t kTotalWidth = []() {
    size_t acc = 0;
    for (size_t i = 0; i < kNumSlots; ++i)
      acc += kWidths[i];
    return acc;
  }();

  static_assert(kTotalWidth <= 64, "OptIndex bitfield exceeds 64 bits; reduce "
                                   "secondary index count or tag count");

  using StorageType = storage_type_t<kTotalWidth>;

  StorageType value{0};

  template <size_t tSlotIndex>
  [[nodiscard]] constexpr size_t get_slot() const noexcept {
    return (static_cast<size_t>(value) >> kOffsets[tSlotIndex]) &
           kMasks[tSlotIndex];
  }

  template <size_t tSlotIndex>
  constexpr void set_slot(size_t encoded) noexcept {
    constexpr size_t offset = kOffsets[tSlotIndex];
    constexpr size_t mask = kMasks[tSlotIndex];
    value = static_cast<StorageType>(
        (static_cast<size_t>(value) & ~(mask << offset)) |
        ((encoded & mask) << offset));
  }
};

template <> struct BitfieldState<> {
  static constexpr size_t kNumSlots = 0;
  static constexpr size_t kTotalWidth = 0;
  template <size_t tSlotIndex>
  [[nodiscard]] constexpr size_t get_slot() const noexcept {
    return 0;
  }

  template <size_t tSlotIndex> constexpr void set_slot(size_t) noexcept {}
};

template <typename... TIndexBys>
using all_tags_tuple_t =
    decltype(std::tuple_cat(std::declval<typename TIndexBys::tags_tuple>()...));

// Helper to locate a type in a tuple. In the following we generally use tuple
// as type list containing secondary indices. Compile error if absent or
// ambiguous.
template <typename T, typename TTuple, size_t tIDX = 0> struct tuple_index_of;

template <typename T, typename THead, typename... TRest, size_t tIDX>
struct tuple_index_of<T, std::tuple<THead, TRest...>, tIDX>
    : std::conditional_t<std::is_same_v<T, THead>,
                         std::integral_constant<size_t, tIDX>,
                         tuple_index_of<T, std::tuple<TRest...>, tIDX + 1>> {};

template <typename T, size_t tIDX>
struct tuple_index_of<T, std::tuple<>, tIDX> {
  static_assert(always_false_v<T>, "Tag not found in any IndexBy declaration");
};

// Find the count of a type T in a tuple. Used to check that T appears exactly
// once in a tuple.
template <typename T, typename TTuple> struct tuple_count_of;

template <typename T>
struct tuple_count_of<T, std::tuple<>> : std::integral_constant<size_t, 0> {};

template <typename T, typename THead, typename... TRest>
struct tuple_count_of<T, std::tuple<THead, TRest...>>
    : std::integral_constant<
          size_t, (std::is_same_v<T, THead> ? 1 : 0) +
                      tuple_count_of<T, std::tuple<TRest...>>::value> {};

// Mapping TTag to the slot index
// Searching each IndexBy's tags_tuple in order.
template <typename TTag, size_t tSlot, typename... TIndexBys>
struct slot_index_impl;

template <typename TTag, size_t tSlot> struct slot_index_impl<TTag, tSlot> {
  static_assert(always_false_v<TTag>,
                "Tag not found in any IndexBy declaration");
};

// Find the "slot" of a tag, or more specifically, map a tag to (the ordinal
// index) its hook, which is the location in the bitfield
template <typename TTag, size_t tSlot, typename THead, typename... TRest>
struct slot_index_impl<TTag, tSlot, THead, TRest...>
    : std::conditional_t<
          (tuple_count_of<TTag, typename THead::tags_tuple>::value >
           0), // found TTag in THead (a hook)'s tag list
          std::integral_constant<size_t, tSlot>, // return current ordinal index
          slot_index_impl<TTag, tSlot + 1, TRest...>> {
}; // recursively go to the next hook

template <typename... TSecondaryIndexBys> struct TagTraits {
  using AllTagsTuple = all_tags_tuple_t<TSecondaryIndexBys...>;

  template <typename TTag>
  static constexpr size_t slot_index_of =
      slot_index_impl<TTag, 0, TSecondaryIndexBys...>::value;

  template <typename TTag>
  static constexpr size_t tag_value_within_slot =
      tuple_index_of<
          TTag, typename std::tuple_element_t<
                    slot_index_of<TTag>,
                    std::tuple<TSecondaryIndexBys...>>::tags_tuple>::value +
      1;

  template <typename TTag>
  static constexpr size_t container_index_of =
      tuple_index_of<TTag, AllTagsTuple>::value;

  // Compile-time uniqueness check: each tag must appear exactly once.
  static constexpr bool tags_unique = []() {
    return []<typename... TTags>(std::tuple<TTags...>) {
      return (... && (tuple_count_of<TTags, AllTagsTuple>::value == 1));
    }(AllTagsTuple{});
  }();
  static_assert(tags_unique,
                "Each tag must appear in exactly one IndexBy declaration");
};

// Helper to repeat type T exactly N times as a std::tuple. Used to generate the
// partitioned indices for a hook
template <typename T, size_t N, typename... Acc>
struct repeat_in_tuple : repeat_in_tuple<T, N - 1, T, Acc...> {};
template <typename T, typename... Acc> struct repeat_in_tuple<T, 0, Acc...> {
  using type = std::tuple<Acc...>;
};

// For one secondary IndexBy at slot index tSlotIdx: K containers of the same
// type (referred with tag in use).
template <typename TSlot, size_t tSlotIdx, typename TIndexBy>
using secondary_slot_containers_t = typename repeat_in_tuple<
    typename HookTrait<typename TIndexBy::hook_type,
                       tSlotIdx + 1>::template Container<TSlot>,
    std::tuple_size_v<typename TIndexBy::tags_tuple>>::type;

// Concatenate all per-hook container tuples.
template <typename TSlot, typename TSeq, typename... TIndexBys>
struct secondary_containers_impl;

template <typename TSlot, size_t... Is, typename... TIndexBys>
struct secondary_containers_impl<TSlot, std::index_sequence<Is...>,
                                 TIndexBys...> {
  using type = decltype(std::tuple_cat(
      std::declval<secondary_slot_containers_t<TSlot, Is, TIndexBys>>()...));
};

template <typename TSlot, typename... TIndexBys>
using secondary_containers_t = typename secondary_containers_impl<
    TSlot, std::index_sequence_for<TIndexBys...>, TIndexBys...>::type;

// Inherit one hook per secondary IndexBy (shared across its K tags).
template <typename TSeq, typename... TIndexBys> struct secondary_hook_inherit;

template <size_t... Is, typename... TIndexBys>
struct secondary_hook_inherit<std::index_sequence<Is...>, TIndexBys...>
    : public HookTrait<typename TIndexBys::hook_type, Is + 1>::Hook... {};

} // namespace detail

// Modify policies
struct ReindexAll {};  // default
struct ReindexNone {}; // pure mutation, zero overhead
template <typename... TTags> struct ReindexOnly {
  using TagSequence = std::tuple<TTags...>; // explicit subset
};

// ===============================================================================
// OptIndex
//
// TAllocator:                    template-template allocator (e.g.
// FixedSizeAllocator<N>::type); must satisfy std::allocator_traits. Both
// allocator contracts are accepted: allocate() may either return nullptr on
// exhaustion (noexcept pools) or throw std::bad_alloc (std allocators) --
// detail::try_allocate normalises both to a nullptr result.

// TPrimaryIndex:                 bare index type (Ordered, Unordered, List) for
// primary key.

// TSecondaryIndexBys..:          IndexBy<Hook, Tags...> declarations for opt-in
// indices.
// ===============================================================================
template <typename TObject, template <typename> typename TAllocator,
          typename TPrimaryIndex, typename... TSecondaryIndexBys>
class OptIndex {
  using PrimaryTrait = detail::HookTrait<TPrimaryIndex, 0>;
  using Traits = detail::TagTraits<TSecondaryIndexBys...>;

  static constexpr size_t kNSecondary = sizeof...(TSecondaryIndexBys);

  struct Slot : TObject,
                PrimaryTrait::Hook,
                detail::secondary_hook_inherit<
                    std::index_sequence_for<TSecondaryIndexBys...>,
                    TSecondaryIndexBys...> {
    detail::BitfieldState<TSecondaryIndexBys...> membership{};
    Slot(auto &&...args) : TObject(std::forward<decltype(args)>(args)...) {}
  };

  using Pool = TAllocator<Slot>;
  using AllocTraits = std::allocator_traits<Pool>;
  using PrimaryContainer = typename PrimaryTrait::template Container<Slot>;
  using SecondaryContainers =
      detail::secondary_containers_t<Slot, TSecondaryIndexBys...>;

  // Helpers to map a tag to its corresponding secondary container.
  template <typename TTag> auto &secondary() {
    return std::get<Traits::template container_index_of<TTag>>(secondaries_);
  }
  template <typename TTag> const auto &secondary() const {
    return std::get<Traits::template container_index_of<TTag>>(secondaries_);
  }

  // The HookTrait for a given secondary slot index (tSlotIdx is 0-based slot,
  // hook index = tSlotIdx+1).
  template <size_t tSlotIdx>
  using SlotHookTrait = detail::HookTrait<
      typename std::tuple_element_t<
          tSlotIdx, std::tuple<TSecondaryIndexBys...>>::hook_type,
      tSlotIdx + 1>;

  Pool pool_;
  PrimaryContainer primary_;
  SecondaryContainers secondaries_;
  size_t size_{0};

public:
  using SlotType = Slot;

  static constexpr size_t slot_size() { return sizeof(Slot); }

  // Iterator for primary, which is anonymous compared with tag-based
  // secondaries
  auto begin() const { return primary_.cbegin(); }
  auto end() const { return primary_.cend(); }
  auto cbegin() const { return primary_.cbegin(); }
  auto cend() const { return primary_.cend(); }
  size_t size() const { return size_; }

  // reserve: forward to primary container when it supports reserve (hash
  // primaries).
  void reserve(size_t n)
    requires requires { primary_.reserve(n); }
  {
    primary_.reserve(n);
  }

  // Primary-key lookup (requires primary to define KeyGetter).
  auto find_primary(auto &&key) const {
    static_assert(
        requires { typename PrimaryTrait::KeyGetter; },
        "FindPrimary requires that the primary index defines a KeyGetter.");
    return primary_.find(std::forward<decltype(key)>(key));
  }

  template <typename TTag> const auto &get() const { return secondary<TTag>(); }
  template <typename TTag> auto &get() { return secondary<TTag>(); }

  // create_all: create and insert into every secondary's first tag (K=1
  // convenience).
  auto create_all(auto &&...args) {
    auto it = create(std::forward<decltype(args)>(args)...);
    if (it == primary_.cend())
      return it;
    bool ok = [&]<size_t... Is>(std::index_sequence<Is...>) {
      return (... && insert_first_tag<Is>(*it));
    }(std::make_index_sequence<kNSecondary>{});
    if (!ok) {
      remove(*it);
      return primary_.cend();
    }
    return it;
  }

  // Create object and index into primary only.
  auto create(auto &&...args) {
    Slot *p = detail::try_allocate(pool_, 1);
    if (!p)
      return primary_.cend();
    // treat noexcept and throwable constructors differently
    if constexpr (std::is_nothrow_constructible_v<Slot, decltype(args)...>) {
      AllocTraits::construct(pool_, p, std::forward<decltype(args)>(args)...);
    } else {
      try {
        AllocTraits::construct(pool_, p, std::forward<decltype(args)>(args)...);
      } catch (...) {
        AllocTraits::deallocate(pool_, p, 1); // no leak
        return primary_.cend();
      }
    }
    auto it = PrimaryTrait::insert(primary_, *p);
    if (it == primary_.end()) {
      AllocTraits::destroy(pool_, p);
      AllocTraits::deallocate(pool_, p, 1);
      return primary_.cend();
    }
    ++size_;
    return typename PrimaryContainer::const_iterator(it);
  }

  // insert<Tag>: link slot into Tag's secondary container.
  // It does nothing if it's already linked (idempotency).
  template <typename TTag> bool insert(const TObject &obj) {
    Slot &slot = to_mutable(static_cast<const Slot &>(obj));
    constexpr size_t kSlotIdx = Traits::template slot_index_of<TTag>;
    constexpr size_t kEncoded = Traits::template tag_value_within_slot<TTag>;
    constexpr size_t kContainer = Traits::template container_index_of<TTag>;
    if (slot.membership.template get_slot<kSlotIdx>() != 0)
      return false;
    auto &c = std::get<kContainer>(secondaries_);
    auto it = SlotHookTrait<kSlotIdx>::insert(c, slot);
    if (it == c.end())
      return false;
    slot.membership.template set_slot<kSlotIdx>(kEncoded);
    return true;
  }

  // unindex<Tag>: unlink from the secondary container for Tag. Returns false if
  // not linked (idempotency).
  template <typename TTag> bool unindex(const Slot &slot) {
    if (!is_tag_linked<TTag>(slot))
      return false;
    constexpr auto kContainer = Traits::template container_index_of<TTag>;
    return deindex_without_check<TTag>(
        std::get<kContainer>(secondaries_).iterator_to(to_mutable(slot)));
  }

  // unindex<Tag>(secondary_iterator): save an iterator finding
  template <typename TTag>
  bool unindex(auto it)
    requires(!std::is_base_of_v<Slot, std::remove_cvref_t<decltype(it)>>)
  {
    if (!is_tag_linked<TTag>(*it))
      return false;
    return deindex_without_check<TTag>(it);
  }

  // remove: erase from all secondaries then primary, destroy slot.
  void remove(const Slot &slot) {
    remove(primary_.iterator_to(to_mutable(static_cast<const Slot &>(slot))));
  }

  // remove(primary_iterator): avoids a second primary iterator_to when the
  // caller already holds the iterator (e.g. from find_primary).
  void remove(auto it)
    requires(!std::is_base_of_v<Slot, std::remove_cvref_t<decltype(it)>>)
  {
    Slot &slot = to_mutable(*it);
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      (..., deindex_slot_if_linked<Is>(slot));
    }(std::make_index_sequence<kNSecondary>{});
    primary_.erase(it);
    AllocTraits::destroy(pool_, &slot);
    AllocTraits::deallocate(pool_, &slot, 1);
    --size_;
  }

  // remove<Tag>(key): find by secondary key then remove.
  template <typename TTag> bool remove(auto &&key) {
    auto &c = secondary<TTag>();
    auto it = c.find(std::forward<decltype(key)>(key));
    if (it == c.end())
      return false;
    remove(*it);
    return true;
  }

  static Slot &to_mutable(const Slot &s) { return const_cast<Slot &>(s); }

  // project<Tag>: iterator into tag's container if slot is there, else cend().
  template <typename TTag> auto project(const TObject &obj) const {
    const Slot &cs = static_cast<const Slot &>(obj);
    const auto &c = secondary<TTag>();
    if (!is_tag_linked<TTag>(cs))
      return c.cend();
    return c.iterator_to(cs);
  }

  // modify<TPolicy>(slot, mutate, rollback): mutate first, then check which
  // linked tags are now out of order; only deindex+reindex those.
  // Returns false and calls rollback if any unique reindex fails.
  template <typename TPolicy>
    requires std::same_as<TPolicy, ReindexAll> ||
             requires { typename TPolicy::TagSequence; }
  bool modify(const Slot &cs, auto &&mutate, auto &&rollback) {
    Slot &s = to_mutable(cs);
    mutate(s);

    auto try_reindex = [&]<typename... TTags>(std::tuple<TTags...>) {
      std::bitset<sizeof...(TTags)> restore_indices{};
      bool has_inconsistency = false;
      {
        size_t pos = 0;
        (..., (has_inconsistency |= restore_indices[pos++] =
                   check_tag_index<TTags>(s)));
      }
      if (!has_inconsistency)
        return true;

      {
        size_t pos = 0;
        (..., (restore_indices[pos++] ? (void)unindex<TTags>(s) : (void)0));
      }
      bool success = true;
      {
        size_t pos = 0;
        (...,
         (success &= restore_indices[pos++] ? try_insert_tag<TTags>(s) : true));
      }
      if (!success) {
        rollback(s);
        size_t pos = 0;
        (..., ([&]() {
           if (restore_indices[pos++]) {
             if (is_tag_linked<TTags>(s))
               unindex<TTags>(s);
             insert<TTags>(s);
           }
         }()));
      }
      return success;
    };
    if constexpr (std::is_same_v<TPolicy, ReindexAll>)
      return try_reindex(typename Traits::AllTagsTuple{});
    else
      return try_reindex(typename TPolicy::TagSequence{});
  }

  // modify<TPolicy>(slot, mutate): auto-rollback via object copy when needed.
  // Decide rollback based on policy and if there is unique tag
  template <typename TPolicy>
    requires std::same_as<TPolicy, ReindexAll> ||
             std::same_as<TPolicy, ReindexNone> ||
             requires { typename TPolicy::TagSequence; }
  bool modify(const Slot &slot, auto &&mutate) {
    auto dispatch_modify = [&]<typename... TTags>(std::tuple<TTags...>) {
      constexpr bool kHasUnique = (... || is_unique_tag<TTags>());
      struct CopyRollback {
        TObject backup;
        explicit CopyRollback(const Slot &s)
            : backup(static_cast<const TObject &>(s)) {}
        void operator()(Slot &s) { static_cast<TObject &>(s) = backup; }
      };
      struct DummyRollback {
        explicit DummyRollback(const Slot &) {}
        void operator()(Slot &) {}
      };
      using RB = std::conditional_t<kHasUnique, CopyRollback, DummyRollback>;
      RB rb{slot};
      return modify<TPolicy>(slot, mutate, rb);
    };
    if constexpr (std::is_same_v<TPolicy, ReindexNone>) {
      mutate(this->to_mutable(slot));
      return true;
    } else if constexpr (std::is_same_v<TPolicy, ReindexAll>) {
      return dispatch_modify(typename Traits::AllTagsTuple{});
    } else {
      return dispatch_modify(typename TPolicy::TagSequence{});
    }
  }

  OptIndex() = default;
  OptIndex(const OptIndex &) = delete;
  OptIndex &operator=(const OptIndex &) = delete;
  OptIndex(OptIndex &&) = delete;
  OptIndex &operator=(OptIndex &&) = delete;

private:
  template <typename TTag> bool deindex_without_check(auto it) {
    constexpr size_t kSlotIdx = Traits::template slot_index_of<TTag>;
    constexpr size_t kContainer = Traits::template container_index_of<TTag>;
    auto &c = std::get<kContainer>(secondaries_);
    if (it == c.end())
      return false;
    Slot &slot = to_mutable(*it);
    c.erase(it);
    slot.membership.template set_slot<kSlotIdx>(0);
    return true;
  }

  template <size_t tSlotIdx, size_t... Is>
  bool apply_to_linked(Slot &slot, size_t encoded, auto &&func,
                       std::index_sequence<Is...>) {

    constexpr size_t kBase = container_offset_for_slot<tSlotIdx>();
    // encoded is 1-based. Container index = kBase + encoded - 1.
    return (... || (encoded == Is + 1
                        ? (func(std::get<kBase + Is>(secondaries_)), true)
                        : false));
  }

  // Number of containers before secondary slot tSlotIdx (prefix sum over K_i).
  template <size_t tSlotIdx>
  static constexpr size_t container_offset_for_slot() {
    size_t off = 0;
    [&]<size_t... Is>(std::index_sequence<Is...>) {
      (..., (Is < tSlotIdx
                 ? (off += std::tuple_size_v<typename std::tuple_element_t<
                        Is, std::tuple<TSecondaryIndexBys...>>::tags_tuple>)
                 : 0));
    }(std::make_index_sequence<kNSecondary>{});
    return off;
  }

  template <size_t tSlotIdx> void deindex_slot_if_linked(Slot &slot) {
    size_t cur = slot.membership.template get_slot<tSlotIdx>();
    if (cur == 0)
      return;
    constexpr size_t kBase = container_offset_for_slot<tSlotIdx>();
    constexpr size_t kK = std::tuple_size_v<typename std::tuple_element_t<
        tSlotIdx, std::tuple<TSecondaryIndexBys...>>::tags_tuple>;
    (void)apply_to_linked<tSlotIdx>(
        slot, cur,
        [&slot](auto &container) {
          container.erase(container.iterator_to(slot)); // unlink the container
          slot.membership.template set_slot<tSlotIdx>(0); // unset the bit field
        },
        std::make_index_sequence<kK>{});
  }

  // Insert into the first (index 0) tag of secondary slot tSlotIdx.
  template <size_t tSlotIdx> bool insert_first_tag(const Slot &cs) {
    using FirstTag = std::tuple_element_t<
        0, typename std::tuple_element_t<
               tSlotIdx, std::tuple<TSecondaryIndexBys...>>::tags_tuple>;
    return try_insert_tag<FirstTag>(to_mutable(cs));
  }

  // True if slot is currently in the container for TTag.
  template <typename TTag> static bool is_tag_linked(const Slot &s) {
    constexpr size_t kSlotIdx = Traits::template slot_index_of<TTag>;
    constexpr size_t kEncoded = Traits::template tag_value_within_slot<TTag>;
    return s.membership.template get_slot<kSlotIdx>() == kEncoded;
  }

  // Returns true if the slot's current position in TTag's container is
  // inconsistent (needs deindex+reindex).
  template <typename TTag> bool check_tag_index(Slot &s) {
    if (!is_tag_linked<TTag>(s))
      return false;
    using Hook = typename std::tuple_element_t<
        Traits::template slot_index_of<TTag>,
        std::tuple<TSecondaryIndexBys...>>::hook_type;
    if constexpr (std::is_same_v<Hook, List>) {
      return false;
    } else if constexpr (!COrderedIndex<Hook>) // unordered (hash), always
                                               // reindex
    {
      return true;
    } else {
      bool consistent = false;
      using KeyGetter = typename Hook::KeyGetter;
      using Compare = typename Hook::Compare;
      auto &container = secondary<TTag>();
      auto it = container.iterator_to(s);
      auto prev_it = it == container.begin() ? container.end() : std::prev(it);
      auto next_it = std::next(it);
      if constexpr (is_unique_tag<TTag>()) {
        consistent = (prev_it == container.end() ||
                      Compare{}(KeyGetter{}(*prev_it), KeyGetter{}(s)));
        consistent &= (next_it == container.end() ||
                       Compare{}(KeyGetter{}(s), KeyGetter{}(*next_it)));
      } else {
        consistent = (prev_it == container.end() ||
                      !Compare{}(KeyGetter{}(s), KeyGetter{}(*prev_it)));
        consistent &= (next_it == container.end() ||
                       !Compare{}(KeyGetter{}(*next_it), KeyGetter{}(s)));
      }
      return !consistent;
    }
  }

  // True if TTag's hook type is a unique index (i.e. not OrderedNonUnique).
  template <typename TTag> static constexpr bool is_unique_tag() {
    using Hook = typename std::tuple_element_t<
        Traits::template slot_index_of<TTag>,
        std::tuple<TSecondaryIndexBys...>>::hook_type;
    return !requires { typename Hook::Compare; } ||
           !std::same_as<Hook, OrderedNonUnique<typename Hook::KeyGetter,
                                                typename Hook::Compare>>;
  }

  // Try to insert into TTag's container; return false if unique-key conflict.
  template <typename TTag> bool try_insert_tag(Slot &slot) {
    constexpr size_t kSlotIdx = Traits::template slot_index_of<TTag>;
    constexpr size_t kEncoded = Traits::template tag_value_within_slot<TTag>;
    constexpr size_t kContainer = Traits::template container_index_of<TTag>;
    auto &c = std::get<kContainer>(secondaries_);
    auto it = SlotHookTrait<kSlotIdx>::insert(c, slot);
    if (it == c.end())
      return false;
    slot.membership.template set_slot<kSlotIdx>(kEncoded);
    return true;
  }
};

template <typename TObject, size_t N, typename TPrimaryIndex,
          typename... TSecondaryIndexBys>
using FixedSizeOptIndex =
    OptIndex<TObject, FixedSizeAllocator<N>::template type, TPrimaryIndex,
             TSecondaryIndexBys...>;

} // namespace optindex
#endif
