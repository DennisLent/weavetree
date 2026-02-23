use weavetree_core::{ActionId, ReturnType, SearchConfig, Tree};
use weavetree_mdp::{DomainSimulator, MdpDomain};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Cell {
    Empty,
    X,
    O,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TicTacToeState {
    board: [Cell; 9],
}

#[derive(Debug, Clone)]
struct TicTacToeDomain {
    start: TicTacToeState,
}

impl Default for TicTacToeDomain {
    fn default() -> Self {
        // Non-obvious opening with one move by each player already played:
        //  X | . | .
        //  . | O | .
        //  . | . | .
        // X to play.
        Self {
            start: TicTacToeState {
                board: [
                    Cell::X,
                    Cell::Empty,
                    Cell::Empty,
                    Cell::Empty,
                    Cell::O,
                    Cell::Empty,
                    Cell::Empty,
                    Cell::Empty,
                    Cell::Empty,
                ],
            },
        }
    }
}

impl TicTacToeDomain {
    fn legal_moves(&self, state: &TicTacToeState) -> Vec<usize> {
        let mut moves = Vec::new();
        for (idx, cell) in state.board.iter().enumerate() {
            if matches!(cell, Cell::Empty) {
                moves.push(idx);
            }
        }
        moves
    }

    fn winner(board: &[Cell; 9]) -> Option<Cell> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for line in LINES {
            let a = board[line[0]];
            let b = board[line[1]];
            let c = board[line[2]];
            if a != Cell::Empty && a == b && b == c {
                return Some(a);
            }
        }
        None
    }

    fn is_full(board: &[Cell; 9]) -> bool {
        board.iter().all(|cell| *cell != Cell::Empty)
    }

    fn render_board(board: &[Cell; 9]) -> String {
        fn to_char(cell: Cell) -> char {
            match cell {
                Cell::Empty => '.',
                Cell::X => 'X',
                Cell::O => 'O',
            }
        }

        format!(
            "{} {} {}\n{} {} {}\n{} {} {}",
            to_char(board[0]),
            to_char(board[1]),
            to_char(board[2]),
            to_char(board[3]),
            to_char(board[4]),
            to_char(board[5]),
            to_char(board[6]),
            to_char(board[7]),
            to_char(board[8]),
        )
    }
}

impl MdpDomain for TicTacToeDomain {
    type State = TicTacToeState;

    fn start_state(&self) -> Self::State {
        self.start.clone()
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        Self::winner(&state.board).is_some() || Self::is_full(&state.board)
    }

    fn num_actions(&self, state: &Self::State) -> usize {
        // `Tree` expects a dense action space [0..num_actions).
        // We map those dense indices back to concrete board cells in `step`.
        if self.is_terminal(state) {
            0
        } else {
            self.legal_moves(state).len()
        }
    }

    fn step(&self, state: &Self::State, action_id: usize, sample: f64) -> (Self::State, f64, bool) {
        // This is the core simulator callback users provide to define dynamics.
        // It can be deterministic or stochastic.
        if self.is_terminal(state) {
            return (state.clone(), 0.0, true);
        }

        let legal_moves = self.legal_moves(state);
        let Some(&x_move) = legal_moves.get(action_id) else {
            return (state.clone(), 0.0, true);
        };

        let mut next = state.clone();
        next.board[x_move] = Cell::X;

        if let Some(winner) = Self::winner(&next.board) {
            let reward = if winner == Cell::X { 1.0 } else { -1.0 };
            return (next, reward, true);
        }
        if Self::is_full(&next.board) {
            return (next, 0.0, true);
        }

        // Opponent plays uniformly at random over legal moves.
        let opp_moves = self.legal_moves(&next);
        if opp_moves.is_empty() {
            return (next, 0.0, true);
        }
        let idx = ((sample * opp_moves.len() as f64) as usize).min(opp_moves.len() - 1);
        let o_move = opp_moves[idx];
        next.board[o_move] = Cell::O;

        if let Some(winner) = Self::winner(&next.board) {
            let reward = if winner == Cell::X { 1.0 } else { -1.0 };
            return (next, reward, true);
        }
        if Self::is_full(&next.board) {
            return (next, 0.0, true);
        }

        (next, 0.0, false)
    }
}

fn main() {
    // Step 1: Build your domain (state/action/reward rules).
    let domain = TicTacToeDomain::default();
    let start_state = domain.start_state();
    let root_moves = domain.legal_moves(&start_state);

    // Step 2: Build a simulator from the domain:
    //    - seeded RNG for reproducibility
    //    - automatic state interning to tree keys
    //    - ready-to-use callbacks for `Tree::run`
    let shared = DomainSimulator::new(domain, 7).into_shared();

    // Step 3 Seed the MCTS tree from start state.
    let mut tree = Tree::new(shared.start_state_key(), shared.root_is_terminal());

    let config = SearchConfig {
        iterations: 800,
        c: 1.0,
        gamma: 1.0,
        max_steps: 6,
        return_type: ReturnType::Discounted,
        fixed_horizon_steps: 6,
    };

    // Step 4: Run search by plugging simulator closures directly into MCTS.
    let run = tree
        .run(
            &config,
            shared.num_actions_fn(),
            shared.step_fn(),
            |_state, _num_actions| ActionId::from(0),
        )
        .expect("MCTS run should succeed");

    // Step 5: Decode best action back to board coordinates for display.
    let best_action = tree
        .best_root_action_by_value()
        .expect("best action lookup should succeed")
        .expect("root should have actions");

    let best_cell = root_moves[best_action.index()];
    let row = best_cell / 3;
    let col = best_cell % 3;

    let mut step = shared.step_fn();
    let (_next_state, sampled_reward, sampled_terminal) =
        step(shared.start_state_key(), best_action);

    println!("Start position:");
    println!("{}", TicTacToeDomain::render_board(&start_state.board));
    println!();
    println!(
        "MCTS completed {} iterations. Best root action index: {}",
        run.iterations_completed,
        best_action.index()
    );
    println!(
        "Chosen move: board cell {} (row {}, col {})",
        best_cell, row, col
    );
    println!(
        "One sampled transition from best move -> reward: {}, terminal: {}",
        sampled_reward, sampled_terminal
    );
}
