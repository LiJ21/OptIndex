#ifndef MULTIMAP
#define MULTIMAP

#include <bitset>
#include <boost/intrusive/link_mode.hpp>
#include <boost/intrusive/list.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/intrusive/unordered_set.hpp>
#include <cstddef>
#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace bi = boost::intrusive;

namespace fastmm {
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

template <typename TObject, size_t tSize> class FixedSizeLifoPool {
public:
  using ObjectType = TObject;
  static constexpr size_t kSize = tSize;
  FixedSizeLifoPool() {
    mem_pool_.reserve(kSize);
    for (auto i : std::views::iota(size_t{0}, kSize)) {
      MemSlot s;
      s.next_free_idx_ = i + 1;
      mem_pool_.push_back(s);
    }
  }

  ObjectType *create(auto &&...args) {
    if (free_head_ == kSize) {
      return nullptr;
    } else {
      auto idx = free_head_;
      free_head_ = mem_pool_[idx].next_free_idx_;
      auto &obj = mem_pool_[idx];
      new (&obj) ObjectType(std::forward<decltype(args)>(args)...);
      --free_count_;
      return reinterpret_cast<ObjectType *>(&obj);
    }
  }

  void remove(ObjectType &obj) {
    auto idx = get_index(obj);
    obj.~ObjectType();
    mem_pool_[idx].next_free_idx_ = free_head_;
    free_head_ = idx;
    ++free_count_;
  }

  bool owns(ObjectType &obj) const { return get_index(obj) < mem_pool_.size(); }

  size_t size() const { return kSize - free_count_; }

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
    for (size_t i = 0; i < mem_pool_.size(); ++i) {
      if (live_mask.test(i)) {
        auto &obj = mem_pool_[i];
        reinterpret_cast<ObjectType &>(obj).~ObjectType();
      }
    }
  }

private:
  size_t get_index(ObjectType &obj) const {
    return static_cast<size_t>(reinterpret_cast<MemSlot *>(&obj) -
                               mem_pool_.data());
  }
  struct alignas(ObjectType) MemSlot {
    union {
      std::byte storage[sizeof(ObjectType)];
      size_t next_free_idx_;
    };
  };
  std::vector<MemSlot> mem_pool_;
  size_t free_head_{0};
  size_t free_count_{kSize};
};

template <size_t tIDX> struct Tag {};

template <typename TIndex, typename TTag> struct Named : public TIndex {
  using Tag = TTag;
};

template <typename TIndex> struct Unnamed;

template <typename TIndex, typename TTag> struct Unnamed<Named<TIndex, TTag>> {
  using Index = TIndex;
  using Tag = TTag;
};

template <typename TIndex> struct Unnamed {
  using Index = TIndex;
};

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

#include <concepts>
#include <cstddef>

template <class...> inline constexpr bool always_false_v = false;

template <std::size_t tIDX, typename TTag, typename... TIndices>
struct GetIndexByTagImpl;

template <std::size_t tIDX, typename TTag, typename TIndex,
          typename... TIndices>
  requires requires { typename TIndex::Tag; } &&
           std::same_as<TTag, typename TIndex::Tag>
struct GetIndexByTagImpl<tIDX, TTag, TIndex, TIndices...> {
  static constexpr std::size_t index = tIDX;
};

template <std::size_t tIDX, typename TTag, typename TIndex,
          typename... TIndices>
struct GetIndexByTagImpl<tIDX, TTag, TIndex, TIndices...> {
  static constexpr std::size_t index =
      GetIndexByTagImpl<tIDX + 1, TTag, TIndices...>::index;
};

template <std::size_t tIDX, typename TTag>
struct GetIndexByTagImpl<tIDX, TTag> {
  static_assert(always_false_v<TTag>, "Tag not found in any index");
};

template <typename TIndexType, size_t tIDX> struct IndexTrait;
template <size_t tIDX> struct IndexTrait<List, tIDX> : List {
  using Hook =
      bi::list_base_hook<bi::link_mode<bi::normal_link>, bi::tag<Tag<tIDX>>>;
  template <typename TObject>
  using Container =
      bi::list<TObject, bi::base_hook<Hook>, bi::constant_time_size<true>>;

  static auto insert(auto &container, auto &obj) {
    container.push_back(obj);
    return container.iterator_to(obj);
  }
};

template <size_t tIDX, typename TKeyGetter, typename TCompare>
struct IndexTrait<Ordered<TKeyGetter, TCompare>, tIDX>
    : Ordered<TKeyGetter, TCompare> {
  using Hook =
      bi::set_base_hook<bi::link_mode<bi::normal_link>, bi::tag<Tag<tIDX>>>;
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
struct IndexTrait<OrderedNonUnique<TKeyGetter, TCompare>, tIDX>
    : OrderedNonUnique<TKeyGetter, TCompare> {
  using Hook =
      bi::set_base_hook<bi::link_mode<bi::normal_link>, bi::tag<Tag<tIDX>>>;
  template <typename TObject>
  using Container =
      bi::multiset<TObject, bi::base_hook<Hook>, bi::constant_time_size<true>,
                   bi::key_of_value<TKeyGetter>, bi::compare<TCompare>>;

  static auto insert(auto &container, auto &obj) {
    return container.insert(obj);
  }
};

template <typename TBucketType, size_t N> struct BucketBase {
  TBucketType buckets_[N];
};

template <size_t tIDX, typename TKeyGetter, typename THash, typename TEqual,
          size_t tBuckets>
struct IndexTrait<Unordered<TKeyGetter, THash, TEqual, tBuckets>, tIDX>
    : public Unordered<TKeyGetter, THash, TEqual, tBuckets> {
  using Hook = bi::unordered_set_base_hook<bi::link_mode<bi::normal_link>,
                                           bi::tag<Tag<tIDX>>>;
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
    if (!ok) {
      return container.end();
    } else {
      return it;
    }
  }
};
} // namespace detail

struct ReindexAll {};  // default
struct ReindexNone {}; // pure mutation, zero overhead
template <size_t... tIDXs> struct ReindexOnly {
  using IndexSequence = std::index_sequence<tIDXs...>;
}; // explicit subset
template <typename... TTags> struct ReindexOnlyByTag {
  using TagSequence = std::tuple<TTags...>;
};

template <typename TObject, template <typename> typename TAllocator,
          typename... TIndices>
class MultiMap {
  using Object = TObject;
  template <typename TSlot> using Allocator = TAllocator<TSlot>;

  template <typename T> struct is_unique_index : std::true_type {};
  template <typename TKeyGetter, typename TCompare>
  struct is_unique_index<OrderedNonUnique<TKeyGetter, TCompare>>
      : std::false_type {};

  static_assert(
      is_unique_index<std::tuple_element_t<0, std::tuple<TIndices...>>>::value,
      "Index 0 must be a unique index.");

  template <class Seq, class... Ts> struct index_zipper;

  template <std::size_t... Is, class... Ts>
  struct index_zipper<std::index_sequence<Is...>, Ts...> {
    using indices =
        std::tuple<detail::IndexTrait<typename Unnamed<Ts>::Index, Is>...>;
  };
  using index_holder =
      typename index_zipper<std::index_sequence_for<TIndices...>,
                            TIndices...>::indices;
  template <typename TTag> static consteval size_t tag_to_index() {
    return detail::GetIndexByTagImpl<0, TTag, TIndices...>::index;
  }

  template <typename TTuple> struct inherit_helper;

  template <typename... Ts>
  struct inherit_helper<std::tuple<Ts...>> : public Ts::Hook... {};

  template <typename TTuple> struct container_type;

  struct Slot : public Object, public inherit_helper<index_holder> {
    Slot(auto &&...args) : Object(std::forward<decltype(args)>(args)...) {}

    bool is_linked(size_t idx) const { return linked_.test(idx); }
    void link(size_t idx) { linked_.set(idx); }
    void unlink(size_t idx) { linked_.reset(idx); }
    std::bitset<sizeof...(TIndices)> linked_{};
  };

  template <typename... Ts> struct container_type<std::tuple<Ts...>> {
    using type = std::tuple<typename Ts::template Container<Slot>...>;
  };
  using ContainerHolder = typename container_type<index_holder>::type;

public:
  static constexpr size_t slot_size() { return sizeof(Slot); }

  template <size_t tIDX> const auto &get() const {
    return std::get<tIDX>(containers_);
  }

  template <typename TTag> const auto &get() const {
    return std::get<tag_to_index<TTag>()>(containers_);
  }

  template <size_t tIDX> auto index(const Slot &s) {
    auto &container = std::get<tIDX>(containers_);
    auto it = std::tuple_element_t<tIDX, index_holder>::insert(container,
                                                               to_mutable(s));

    if (it == container.end())
      return container.cend();
    it->link(tIDX);
    return container.iterator_to(const_cast<const Slot &>(*it));
  }

  template <typename TTag> auto index(const Slot &s) {
    return index<tag_to_index<TTag>()>(s);
  }

  auto iterator_to(auto &&s) {
    auto &primary = std::get<0>(containers_);
    return primary.iterator_to(std::forward<decltype(s)>(s));
  }

  template <bool tAddAllIndices = true> const auto insert(auto &&...args) {
    auto *pobj = allocator_.create(std::forward<decltype(args)>(args)...);
    if (pobj == nullptr)
      return this->cend();

    auto it = index<0>(*pobj);
    if (it == this->cend()) {
      allocator_.remove(*pobj);
      return this->cend();
    }
    this->to_mutable(*it).link(0);

    if constexpr (tAddAllIndices) {
      bool failed = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return (... || (index<Is + 1>(*pobj) == get<Is + 1>().cend()));
      }(std::make_index_sequence<sizeof...(TIndices) - 1>{});
      if (failed) {
        auto it = this->iterator_to(*pobj);
        if (it != this->end())
          deindex<0>(it);
        return this->cend();
      }
    }

    return it;
  }

  const auto find_primary(auto &&key) const {
    static_assert(
        requires {
          typename std::tuple_element_t<0, std::tuple<TIndices...>>::KeyGetter;
        }, "FindPrimary requires that the primary index defines a KeyGetter.");
    auto &container = this->get<0>();
    auto it = container.find(std::forward<decltype(key)>(key));
    return it;
  }

  template <size_t tIDX> const auto project(const Slot &slot) const {
    if (!slot.is_linked(tIDX))
      return get<tIDX>().cend();
    return get<tIDX>().iterator_to(slot);
  }

  template <typename TTag> const auto project(const Slot &slot) const {
    return project<tag_to_index<TTag>()>(slot);
  }

  template <typename... TTags>
    requires(... && requires { tag_to_index<TTags>(); })
  bool modify(const Slot &slot, auto &&mutate, auto &&rollback) {
    return modify<tag_to_index<TTags>()...>(slot, mutate, rollback);
  }

  template <size_t... tIDXs>
  bool modify(const Slot &slot, auto &&mutate, auto &&rollback) {
    Slot &s = to_mutable(slot);
    static_assert(
        !(... || (tIDXs == 0)),
        "Primary index (0) cannot be reindexed. Use remove() instead.");
    auto reindex = [&](Slot &slot) {
      auto it = this->iterator_to(slot);
      mutate(slot);
      std::bitset<sizeof...(tIDXs)> restore_indices{};

      auto check_index = [&]<size_t tIDX>(Slot &s) {
        if (!s.is_linked(tIDX))
          return false;
        auto &container = std::get<tIDX>(containers_);
        auto it = container.iterator_to(s);
        using Index = std::tuple_element_t<tIDX, std::tuple<TIndices...>>;
        bool consistent = false;
        if constexpr (COrderedIndex<Index>) {
          using KeyGetter =
              typename std::tuple_element_t<tIDX, index_holder>::KeyGetter;
          using Compare =
              typename std::tuple_element_t<tIDX, index_holder>::Compare;
          auto prev_it =
              it == container.begin() ? container.end() : std::prev(it);

          auto next_it = std::next(it);
          if constexpr (is_unique_index<Index>::value) {
            consistent = prev_it == container.end() ||
                         Compare{}(KeyGetter{}(*prev_it), KeyGetter{}(*it));
            consistent &= next_it == container.end() ||
                          Compare{}(KeyGetter{}(*it), KeyGetter{}(*next_it));
          } else {
            consistent = prev_it == container.end() ||
                         !Compare{}(KeyGetter{}(*it), KeyGetter{}(*prev_it));
            consistent &= next_it == container.end() ||
                          !Compare{}(KeyGetter{}(*next_it), KeyGetter{}(*it));
          }
        }
        return !consistent;
      };

      bool has_inconsistency = false;
      {
        size_t pos = 0;
        (..., (has_inconsistency |= restore_indices[pos++] =
                   check_index.template operator()<tIDXs>(s)));
      }
      if (!has_inconsistency) {
        return true;
      }

      {
        size_t pos = 0;
        (..., (restore_indices[pos++] && this->deindex<tIDXs>(it)));
      }

      bool success = true;
      {
        size_t pos = 0;
        (..., (success &= restore_indices[pos++]
                              ? this->index<tIDXs>(slot) != get<tIDXs>().end()
                              : true));
      }

      if (!success) {
        rollback(slot);
        size_t pos = 0;
        (..., ([&]() {
           if (restore_indices[pos++]) {
             if (slot.is_linked(tIDXs))
               this->deindex<tIDXs>(it);
             this->index<tIDXs>(slot);
           }
         }()));
      }
      return success;
    };
    return reindex(s);
  }

  template <typename TPolicy = ReindexAll>
    requires std::same_as<TPolicy, ReindexAll> ||
             std::same_as<TPolicy, ReindexNone> ||
             requires { typename TPolicy::IndexSequence; } ||
             requires { typename TPolicy::TagSequence; }
  bool modify(const Slot &slot, auto &&mutate) {
    Slot &s = to_mutable(slot);
    struct CopyRollback {
      CopyRollback(const Slot &s)
          : backup_slot(static_cast<const Object &>(s)) {}
      void operator()(Slot &s) { static_cast<Object &>(s) = backup_slot; }
      const Object backup_slot;
    };
    struct DummyRollback {
      DummyRollback(const Slot &) {}
      void operator()(Slot &) {}
    };
    constexpr size_t kNoOffset = 0;
    constexpr size_t kOffsetNoPrimary = 1;
    auto modify_with_rollback = [&]<size_t tOffset, size_t... tIDXs>(
                                    Slot &s, std::index_sequence<tIDXs...>) {
      constexpr bool has_unique =
          (... || (is_unique_index<std::tuple_element_t<
                       tIDXs + tOffset, std::tuple<TIndices...>>>::value));
      using Rollback =
          std::conditional_t<has_unique, CopyRollback, DummyRollback>;
      Rollback roll_back(slot);
      return this->modify<tOffset + tIDXs...>(s, mutate, roll_back);
    };
    if constexpr (std::same_as<TPolicy, ReindexAll>) {
      return modify_with_rollback.template operator()<kOffsetNoPrimary>(
          s, std::make_index_sequence<sizeof...(TIndices) - 1>{});
    } else if constexpr (std::same_as<TPolicy, ReindexNone>) {
      mutate(s);
      return true;
    } else if constexpr (requires { typename TPolicy::IndexSequence; }) {
      static_assert(
          ![]<size_t... tIDXs>(std::index_sequence<tIDXs...>) {
            return (... || (tIDXs == 0));
          }(typename TPolicy::IndexSequence{}),
          "Primary index (0) cannot be reindexed.");
      return modify_with_rollback.template operator()<kNoOffset>(
          s, typename TPolicy::IndexSequence{});
    } else if constexpr (requires { typename TPolicy::TagSequence; }) {
      auto tag_to_index_sequence = []<typename... TTags>(std::tuple<TTags...>) {
        return std::index_sequence<tag_to_index<TTags>()...>{};
      };
      return modify_with_rollback.template operator()<kNoOffset>(
          s, tag_to_index_sequence(typename TPolicy::TagSequence{}));
    }
  }

  auto &to_mutable(const Slot &s) { return const_cast<Slot &>(s); }

  template <size_t tIDX> bool unindex(const Slot &s) {
    static_assert(tIDX != 0,
                  "Cannot unindex primary index. Use remove() instead.");
    return deindex<tIDX>(this->iterator_to(to_mutable(s)));
  }

  template <typename TTag> bool unindex(const Slot &s) {
    return unindex<tag_to_index<TTag>()>(s);
  }

  bool remove(const Slot &s) {
    return deindex<0>(this->iterator_to(to_mutable(s)));
  }

  template <size_t tIDX = 0> bool remove(const auto &key) {
    static_assert(
        is_unique_index<
            std::tuple_element_t<tIDX, std::tuple<TIndices...>>>::value,
        "Remove by key requires a unique index. ");

    auto sit = this->get_mutable<tIDX>().find(key);
    if (sit == this->get_mutable<tIDX>().end())
      return false;
    if constexpr (tIDX == 0) {
      deindex<0>(sit);
    } else {
      auto it = this->iterator_to(*sit);
      deindex<0>(it);
    }
    return true;
  }

  template <typename TTag> bool remove(auto &&key) {
    return remove<tag_to_index<TTag>()>(std::forward<decltype(key)>(key));
  }

  auto cbegin() const { return get<0>().cbegin(); }
  auto cend() const { return get<0>().cend(); }
  auto begin() const { return this->cbegin(); }
  auto end() const { return this->cend(); }

  auto size() const { return allocator_.size(); }

  template <typename... TArgs>
    requires(sizeof...(TArgs) != 1 ||
             !std::same_as<
                 std::decay_t<std::tuple_element_t<0, std::tuple<TArgs...>>>,
                 std::initializer_list<TObject>>)
  MultiMap(TArgs &&...args)
      : allocator_(std::forward<decltype(args)>(args)...) {}

  MultiMap() = default;

  MultiMap(std::initializer_list<TObject> init)
    requires std::is_default_constructible_v<Allocator<Slot>>
  {
    for (const auto &obj : init)
      insert(obj);
  }
  MultiMap(const MultiMap &) = delete;
  MultiMap &operator=(const MultiMap &) = delete;
  MultiMap(MultiMap &&) = delete;
  MultiMap &operator=(MultiMap &&) = delete;

private:
  template <size_t tIDX> auto &get_mutable() {
    return std::get<tIDX>(containers_);
  }

  template <size_t tIDX> bool erase_from_index(auto &&it) {
    if (!it->is_linked(tIDX))
      return false;
    auto &container = std::get<tIDX>(containers_);
    if constexpr (tIDX == 0) {
      container.erase(it);
    } else {
      container.erase(container.iterator_to(*it));
      it->unlink(tIDX);
    }
    return true;
  }

  template <size_t tIDX> bool deindex(auto &&it) {
    if constexpr (tIDX == 0) {
      auto &primary = std::get<0>(containers_);
      if (it == primary.end())
        return false;
      Slot &slot = to_mutable(*it);
      [&]<size_t... Is>(std::index_sequence<Is...>) {
        (..., erase_from_index<sizeof...(TIndices) - 1 - Is>(it));
      }(std::make_index_sequence<sizeof...(TIndices) - 1>{});
      primary.erase(it);
      allocator_.remove(slot);
      return true;
    } else {
      return erase_from_index<tIDX>(std::forward<decltype(it)>(it));
    }
  }

  Allocator<Slot> allocator_;
  ContainerHolder containers_;
};

namespace detail {
template <typename TObject, size_t tSize, typename... TIndices>
class FixedSizeMultiMapTrait {
private:
  template <typename T> using PoolWithSize = FixedSizeLifoPool<T, tSize>;

public:
  using type = MultiMap<TObject, PoolWithSize, TIndices...>;
};
} // namespace detail

template <typename TObject, size_t tSize, typename... TIndices>
using FixedSizeMultiMap =
    typename detail::FixedSizeMultiMapTrait<TObject, tSize, TIndices...>::type;
} // namespace fastmm
#endif
