// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This short sample program demonstrates STX B+ Tree's API.
 */

#include "../stx/btree.h"

#include <iostream>
#include <random>

#define KEY_TYPE int
#define PAYLOAD_TYPE int

typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> LEAF_ENTRY_TYPE;

int main(int, char**) {
  // Create some synthetic data: keys are dense integers between 0 and 99, and
  // payloads are random values
  const int num_keys = 100;
  std::vector<std::pair<KEY_TYPE, PAYLOAD_TYPE> > pairs(num_keys);
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_int_distribution<PAYLOAD_TYPE> dis;
  for (int i = 0; i < num_keys; i++) {
    pairs[i].first = i;
    pairs[i].second = dis(gen);
  }

  stx::btree<KEY_TYPE, PAYLOAD_TYPE> index;

  // Bulk load the keys
  index.bulk_load(pairs.begin(), pairs.end());
  std::cout << "num of keys: " << index.size() << std::endl;
  
  std::vector<LEAF_ENTRY_TYPE> v_res;
  KEY_TYPE test_min_key = 1;
  KEY_TYPE test_max_key = 20;
  v_res = index.query_range(test_min_key, test_max_key);
  std::cout << "num of res: " << v_res.size() << std::endl;
}