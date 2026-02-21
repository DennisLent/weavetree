use std::cell::RefCell;
use std::path::PathBuf;

use weavetree_core::{ActionId, ReturnType, SearchConfig, StateKey as CoreStateKey, Tree};
use weavetree_mdp::{MdpSimulator, StateKey, compile_yaml};

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/weavetree-mdp/examples/sample.mdp.yaml"));

    let compiled = compile_yaml(&path).expect("failed to compile MDP YAML");
    let start = compiled.start();
    let simulator = RefCell::new(MdpSimulator::new(compiled, 12345));

    let mut tree = Tree::new(CoreStateKey::from(start.index() as u64), false);
    let config = SearchConfig {
        iterations: 100,
        c: 0.0,
        gamma: 1.0,
        max_steps: 4,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 4,
    };

    let run = tree
        .run(
            &config,
            |state| {
                simulator
                    .borrow()
                    .num_actions(StateKey::from(state.value() as usize))
            },
            |state, action| {
                let (next, reward, terminal) = simulator
                    .borrow_mut()
                    .step(StateKey::from(state.value() as usize), action.index());
                (CoreStateKey::from(next.index() as u64), reward, terminal)
            },
            |_state, _num_actions| ActionId::from(0),
        )
        .expect("MCTS run failed");

    let best = tree
        .best_root_action_by_value()
        .expect("lookup failed")
        .expect("root has no actions");

    println!(
        "best_root_action={} iterations={}",
        best.index(),
        run.iterations_completed
    );
    println!("average_total_return={:.6}", run.average_total_return);
}
