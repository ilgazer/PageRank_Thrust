#include <cstdio>
#include <iostream>
#include <ostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <array>
#include <chrono>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

static std::unordered_map<std::string, int> site_ids_by_ref;
static std::vector<std::string> site_refs;
static thrust::host_vector<int> h_num_outgoing_edges;

static thrust::host_vector<int> h_froms;
static thrust::host_vector<int> h_tos;

int get_site_id(const std::string &name)
{
    auto result = site_ids_by_ref.find(name);
    if (result == site_ids_by_ref.end())
    {
        int site = site_ids_by_ref.size();
        site_ids_by_ref.insert({name, site});
        site_refs.push_back(name);
        h_num_outgoing_edges.push_back(0);
        return site;
    }
    else
    {
        return result->second;
    }
}

struct site_pair
{
    int site_id;
    float score;
};

std::ostream &operator<<(std::ostream &os,
                         site_pair &pair)
{
    // Printing all the elements
    // using <<
    os << "[site_id=" << pair.site_id << ", score=" << pair.score << "]";
    return os;
}
class top_five_array
{
    site_pair arr[5];

public:
    top_five_array() : arr{site_pair{-1, -1},
                           site_pair{-1, -1},
                           site_pair{-1, -1},
                           site_pair{-1, -1},
                           site_pair{-1, -1}} {}

    __host__ __device__ top_five_array(site_pair pair) : arr{pair,
                                                             site_pair{-1, -1},
                                                             site_pair{-1, -1},
                                                             site_pair{-1, -1},
                                                             site_pair{-1, -1}} {}

    top_five_array(site_pair a, site_pair b, site_pair c, site_pair d, site_pair e) : arr{a, b, c, d, e} {}

    inline site_pair &operator[](size_t i)
    {
        return arr[i];
    }

    auto begin()
    {
        return &(arr[0]);
    }

    auto end()
    {
        return &(arr[5]);
    }
};

struct merge_arrays
{
    __host__ __device__ top_five_array operator()(top_five_array a, top_five_array b)
    {
        top_five_array result;

        size_t a_index = 0;
        size_t b_index = 0;
        for (size_t i = 0; i < 5; i++)
        {
            int is_b_larger = a[a_index].score < b[b_index].score;
            result[i] = is_b_larger ? b[b_index] : a[a_index];
            b_index += is_b_larger;
            a_index += (1 - is_b_larger);
        }
        return result;
    }
};

struct absolute_value
{
    __host__ __device__ float operator()(const float x) const
    {
        return x < 0 ? -x : x;
    }
};

struct multiply
{
    __host__ __device__ float operator()(thrust::tuple<float, float> t)
    {
        return thrust::get<0>(t) * thrust::get<1>(t);
    }
};

struct subtract
{
    __host__ __device__ float operator()(thrust::tuple<float, float> t)
    {
        return thrust::get<0>(t) - thrust::get<1>(t);
    }
};

float alpha = 0.2;

int main(int argc, char *argv[])
{
    std::ifstream txt(argv[1]);

    std::string first_token, second_token;
    while (txt >> first_token)
    {
        txt >> second_token;

        int from = get_site_id(first_token);
        int to = get_site_id(second_token);

        h_num_outgoing_edges[from]++;
        h_froms.push_back(from);
        h_tos.push_back(to);
    }

    thrust::device_vector<int> d_num_outgoing_edges = h_num_outgoing_edges;
    thrust::device_vector<int> d_col_indices = h_froms;
    thrust::device_vector<int> d_row_indices = h_tos;

    thrust::sort_by_key(d_row_indices.begin(), d_row_indices.end(), d_col_indices.begin());

    //std::cout << print(d_row_indices) << std::endl;
    //std::cout << print(d_col_indices) << std::endl;

    thrust::device_vector<float> d_values(d_row_indices.size());

    thrust::constant_iterator<float> iter(1);
    thrust::transform(iter, iter + d_row_indices.size(),
                      thrust::make_permutation_iterator(d_num_outgoing_edges.begin(), d_col_indices.begin()),
                      d_values.begin(),
                      thrust::divides());

    //std::cout << print(d_values) << std::endl;

    thrust::device_vector<float> r(h_num_outgoing_edges.size());
    thrust::device_vector<float> r_next(h_num_outgoing_edges.size());

    thrust::fill(r.begin(), r.end(), 1.0);
    thrust::fill(r_next.begin(), r_next.end(), 0.8);

    thrust::device_vector<int> d_conn_rows(d_row_indices.size());
    thrust::unique_copy(d_row_indices.begin(), d_row_indices.end(), d_conn_rows.begin());

    float sigma = 10;
    int sigma_count = 0;

    auto t1 = high_resolution_clock::now();

    while (sigma > 1e-6)
    {
        thrust::permutation_iterator r_col_begin(r.begin(), d_col_indices.begin());
        thrust::permutation_iterator r_next_rows_iter(r_next.begin(), d_conn_rows.begin());

        thrust::device_vector<float> sum_elems(d_values.size());

        thrust::transform(d_values.begin(), d_values.end(), r_col_begin, sum_elems.begin(), thrust::multiplies());

        //std::cout << "sum_elems=" << print(sum_elems) << std::endl;

        thrust::reduce_by_key(
            d_row_indices.begin(),
            d_row_indices.end(),
            sum_elems.begin(),
            thrust::make_discard_iterator(),
            thrust::make_transform_output_iterator(r_next_rows_iter, [](float r_val)
                                                   { return r_val * alpha + (1 - alpha); }));

        //std::cout << "r_next_rows_iter=" << make_print(r_next_rows_iter, r_next_rows_iter + 4) << std::endl;
        //std::cout << "r_next=" << print(r_next) << std::endl;

        thrust::transform_iterator deltas_begin(
            thrust::make_zip_iterator(r.begin(), r_next.begin()),
            subtract());

        thrust::transform_iterator deltas_end(
            thrust::make_zip_iterator(r.end(), r_next.end()),
            subtract());

        sigma = thrust::transform_reduce(deltas_begin, deltas_end, absolute_value(), 0.0, thrust::plus());
        sigma_count++;
        //std::cout << "sigma_" << sigma_count << ": " << sigma << std::endl;

        r.swap(r_next);
        thrust::fill(r_next.begin(), r_next.end(), 0.8);
    }

    //std::cout << print(r) << std::endl;

    thrust::counting_iterator<int> ids(0);
    top_five_array top_five = thrust::transform_reduce(
        thrust::make_zip_iterator(ids, r.begin()),
        thrust::make_zip_iterator(ids + r.size(), r.end()),
        [](auto t)
        { return top_five_array(site_pair{thrust::get<0>(t), thrust::get<1>(t)}); },
        top_five_array(),
        merge_arrays());

    auto t2 = high_resolution_clock::now();

    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << "Duration:" << ms_int.count() << std::endl;

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << site_refs[top_five[i].site_id] << ": " << top_five[i].score << std::endl;
    }

    return 0;
}