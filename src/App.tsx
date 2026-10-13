import { useSyncExternalStore } from "react";
import { Coordinator } from "./engine/coordinator";
import { PlayPanel } from "./components/PlayPanel";
import { TrainPanel } from "./components/TrainPanel";

const coord = new Coordinator();

export default function App() {
  const snap = useSyncExternalStore(coord.subscribe, coord.getSnapshot);

  return (
    <div className="min-h-screen bg-[#fbfbf9] text-neutral-900">
      <div className="max-w-6xl mx-auto px-5 py-8">
        <header className="mb-8 border-b border-neutral-200 pb-5">
          <h1 className="text-xl font-semibold tracking-tight">
            az4 <span className="font-normal text-neutral-400">— AlphaZero for Connect Four, in your browser</span>
          </h1>
          <p className="text-xs text-neutral-500 mt-2 max-w-3xl leading-relaxed">
            No pretrained weights, no libraries. A policy/value network learns from scratch using only self-play guided by
            PUCT Monte-Carlo tree search. Self-play runs across a pool of Web Workers, a trainer worker learns from the
            replay buffer, and every generation plays rated matches against earlier ones. Press <b>start self-play</b>,
            then play against any checkpoint while it trains.
          </p>
        </header>

        <main className="grid lg:grid-cols-[auto_1fr] gap-x-12 gap-y-10">
          <div className="lg:w-[380px]">
            <PlayPanel coord={coord} snap={snap} />
          </div>
          <div className="min-w-0">
            <TrainPanel coord={coord} snap={snap} />
          </div>
        </main>

        <section className="mt-12 border-t border-neutral-200 pt-6 grid md:grid-cols-4 gap-6 text-[11.5px] leading-relaxed text-neutral-600">
          <div>
            <h3 className="label">search</h3>
            PUCT: pick the move with the highest Q + c·P·√N / (1+n), where P comes from the network's policy and leaves are
            scored by its value head. No random rollouts. Dirichlet noise (α=1, ε=.25) at the root during self-play.
          </div>
          <div>
            <h3 className="label">network</h3>
            MLP 84→128→128 with a 7-way policy head and a tanh value head (~28k params), written in plain TypeScript
            with hand-written backprop and AdamW. Input is the board from the side-to-move's view.
          </div>
          <div>
            <h3 className="label">loop</h3>
            Each generation: N self-play games (T=1 for the first 10 plies) → (s, π, z) samples, mirrored left-right →
            60k ring buffer → minibatch SGD on CE(π, p) + (z − v)² → new weights go straight back to the workers.
          </div>
          <div>
            <h3 className="label">Elo</h3>
            Gen 0 (random weights) is fixed at 0. Each new generation plays the previous one and the one from 4 gens
            earlier, and its rating is the maximum-likelihood fit against their frozen ratings. Pure rollout MCTS and a
            random player are tracked as outside baselines.
          </div>
        </section>

        <footer className="mt-10 text-[11px] text-neutral-400">
          keys: 1–7 drop · u undo · n new game. Latest weights are saved in localStorage.
        </footer>
      </div>
    </div>
  );
}
