use std::collections::HashMap;

use proptest::prelude::*;

use crate::tree::{
    ids::{NodeId, StateKey},
    outcomes::OutcomeSet,
};

proptest! {
    #[test]
    fn outcome_set_bookkeeping_matches_observed_frequencies(sequence in proptest::collection::vec(0u8..8u8, 1..128)) {
        let mut set = OutcomeSet::new();
        let mut expected_counts: HashMap<u64, u64> = HashMap::new();

        for (idx, raw_key) in sequence.iter().copied().enumerate() {
            let state_key = StateKey::from(raw_key as u64);
            let entry = expected_counts.entry(state_key.value()).or_insert(0);
            *entry += 1;

            if set.get_child_for(state_key).is_some() {
                let child = set.increment_outcome(state_key);
                prop_assert!(child.is_some());
            } else {
                let inserted = set.insert_outcome(state_key, NodeId::from(idx));
                prop_assert!(inserted.is_some());
            }
        }

        prop_assert_eq!(set.len(), expected_counts.len());

        for (raw_key, count) in expected_counts {
            let state_key = StateKey::from(raw_key);
            prop_assert_eq!(set.count_for(state_key), Some(count));
        }
    }
}
