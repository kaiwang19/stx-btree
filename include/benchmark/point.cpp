#include "../stx/btree.h"
#include "../alex/alex.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include "flags.h"
#include "utils.h"


// #define KEY_TYPE double
// #define PAYLOAD_TYPE double

#define KEY_TYPE uint64_t 
#define PAYLOAD_TYPE uint64_t 

int main(int argc, char *argv[])
{
      auto flags = parse_flags(argc, argv);
      std::string keys_file_path = get_required(flags, "keys_file");
      std::string keys_file_type = get_required(flags, "keys_file_type");
      auto init_num_keys = stoi(get_required(flags, "init_num_keys"));
      auto batch_size = stoi(get_required(flags, "batch_size"));
      std::string lookup_distribution =
          get_with_default(flags, "lookup_distribution", "uniform");

      // Read keys from file
      auto keys = new KEY_TYPE[init_num_keys];
      if (keys_file_type == "binary")
      {
            load_binary_data(keys, init_num_keys, keys_file_path);
      }
      else if (keys_file_type == "text")
      {
            load_text_data(keys, init_num_keys, keys_file_path);
      }
      else
      {
            std::cerr << "--keys_file_type must be either 'binary' or 'text'"
                      << std::endl;
            return 1;
      }

      std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE>> pairs(init_num_keys);
      auto values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_num_keys];
      std::mt19937_64 gen_payload(std::random_device{}());
      for (int i = 0; i < init_num_keys; i++)
      {
            pairs[i].first = keys[i];
            pairs[i].second = static_cast<PAYLOAD_TYPE>(gen_payload());
            values[i].first = keys[i];
            values[i].second = static_cast<PAYLOAD_TYPE>(gen_payload());
      }
      std::sort(pairs.begin(), pairs.end(), [](auto const &a, auto const &b) { return a.first < b.first; }); 
      std::sort(values, values + init_num_keys, [](auto const &a, auto const &b) { return a.first < b.first; }); 

      // Run workload
      PAYLOAD_TYPE sum = 0;
      std::cout << std::fixed; // scientific
      std::cout << std::setprecision(4);

      KEY_TYPE *lookup_keys = nullptr;
      if (lookup_distribution == "uniform")
      {
            lookup_keys = get_search_keys(keys, init_num_keys, batch_size);
      }
      else if (lookup_distribution == "zipf")
      {
            lookup_keys = get_search_keys_zipf(keys, init_num_keys, batch_size);
      }
      else
      {
            std::cerr << "--lookup_distribution must be either 'uniform' or 'zipf'"
                      << std::endl;
            return 1;
      }


      // Create STX B+ Tree and bulk load
      stx::btree<KEY_TYPE, PAYLOAD_TYPE> index_bplus;
      index_bplus.bulk_load(pairs.begin(), pairs.end());
      index_bplus.print_stat();
      index_bplus.print_root_depth();

      // Create ALEX and bulk load
      alex::Alex<KEY_TYPE, PAYLOAD_TYPE> index_alex;
      index_alex.bulk_load(values, init_num_keys);
      index_alex.print_stat();
      index_alex.print_all_data_nodes();

      auto lookups_start_time =std::chrono::high_resolution_clock::now();

      for (int i = 0; i < batch_size; i++)
      {
            KEY_TYPE key = lookup_keys[i];
            // PAYLOAD_TYPE *payload = index_bplus.get_payload(key);
            stx::btree<KEY_TYPE, PAYLOAD_TYPE>::iterator payload = index_bplus.find(key);
            if (payload.data())
            {
                  sum += payload.data();
            }
      }
      auto lookups_end_time = std::chrono::high_resolution_clock::now();
      double batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lookups_end_time - lookups_start_time)
                    .count();
      std::cout << "point query time (B+):   \t"
                << (batch_lookup_time / batch_size)
                << " ns/query"
                << std::endl;


      lookups_start_time =std::chrono::high_resolution_clock::now();
      for (int j = 0; j < batch_size; j++)
      {
            KEY_TYPE key = lookup_keys[j];
            PAYLOAD_TYPE *payload = index_alex.get_payload(key);
            if (payload)
            {
                  sum += *payload;
            }
      }
      lookups_end_time = std::chrono::high_resolution_clock::now();
      batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lookups_end_time - lookups_start_time)
                    .count();
      std::cout << "point query time (ALEX):\t"
                << (batch_lookup_time / batch_size)
                << " ns/query"
                << std::endl;


      

      delete[] lookup_keys;
      delete[] keys;
      pairs.clear();

      return 0;
}
