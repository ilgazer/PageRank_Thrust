#include <cstdio>
#include <iostream>
#include <ostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <array>

template <typename A>
struct fake_iterable
{
    A &_begin;
    A &_end;

    fake_iterable(A &__begin, A &__end) : _begin(__begin), _end(__end) {}

    A &begin() { return _begin; }

    A &end()
    {
        return _end;
    }
};

template <typename R>
struct print
{
    R &r;
    explicit print(R &_r) : r(_r) {}
};

template <typename A>
print<fake_iterable<A>> make_print(A &begin, A &end) { return print(*(new fake_iterable(begin, end))); }

template <typename A>
print<fake_iterable<A>> make_print(A begin, A end) { return print(*(new fake_iterable(begin, end))); }


template <typename R>
std::ostream &operator<<(std::ostream &os,
                         const print<R> &vector)
{
    // Printing all the elements
    // using <<
    os << "[";
    for (auto element : vector.r)
    {
        os << element << ", ";
    }
    os << "]";
    return os;
}

