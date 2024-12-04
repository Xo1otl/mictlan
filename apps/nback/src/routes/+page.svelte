<script lang="ts">
  import TaskInfo from "../game/TaskInfo.svelte";
  import StimulusGrid from "../game/StimulusGrid.svelte";
  import ResponseOptions from "../game/ResponseOptions.svelte";
  import Scoreboard from "../game/Scoreboard.svelte";
  import { NewManager } from "../game/game";
  import type { Queue, Stimulus } from "../game/game";
  import type { Config, ScoreRecords as ScoreRecord } from "../game/game";

  const config: Config = $state({
    n: 2,
    delay: 4000,
    totalQuestions: 16,
  });

  let scoreRecord: ScoreRecord = $state({
    coordinates: { matches: 0, total: 0 },
    character: { matches: 0, total: 0 },
    color: { matches: 0, total: 0 },
    shape: { matches: 0, total: 0 },
    total: { matches: 0, total: 0 },
    update: (tryResult, actualResult) => {
      console.log(tryResult.shape, actualResult.shape);
      if (!tryResult) return;

      scoreRecord.total.total++;
      if (tryResult.coordinates === actualResult.coordinates) {
        scoreRecord.coordinates.matches++;
      }
      scoreRecord.coordinates.total++;

      if (tryResult.character === actualResult.character) {
        scoreRecord.character.matches++;
      }
      scoreRecord.character.total++;

      if (tryResult.color === actualResult.color) {
        scoreRecord.color.matches++;
      }
      scoreRecord.color.total++;

      if (tryResult.shape === actualResult.shape) {
        scoreRecord.shape.matches++;
      }
      scoreRecord.shape.total++;

      if (
        tryResult.coordinates === actualResult.coordinates &&
        tryResult.character === actualResult.character &&
        tryResult.color === actualResult.color &&
        tryResult.shape === actualResult.shape
      ) {
        scoreRecord.total.matches++;
      }
    },
  });

  let current: Stimulus | undefined = $state(undefined);
  let taskStarted = $state(false);

  function createStack(capacity: number): Queue {
    let items: Stimulus[] = $state([]);
    const full = () => {
      return items.length >= capacity;
    };

    const push = (stimulus: Stimulus) => {
      if (full()) {
        items = [...items.slice(1), stimulus];
      } else {
        items = [...items, stimulus];
      }
      current = stimulus;
    };

    const pop = () => {
      if (items.length === 0) {
        return undefined;
      }
      const poppedItem = items[0];
      return poppedItem;
    };

    return {
      isFull: full,
      push,
      pop,
    };
  }

  const stack = createStack(config.n);
  const manager = NewManager(config, scoreRecord, stack);
</script>

<div
  class="flex w-full flex-col items-center justify-center h-screen font-sans"
>
  <h1 class="text-3xl font-bold mb-8 text-center">N-Back Task</h1>
  <TaskInfo n={config.n} totalQuestions={config.totalQuestions} />
  <div class="text-2xl mb-4">
    <span>Delay: {config.delay}ms</span>
  </div>
  <StimulusGrid stimulus={current} />
  <ResponseOptions {manager} />
  <Scoreboard {scoreRecord} />
  <button
    class="px-4 py-2 rounded-md bg-blue-500 text-white mt-5"
    aria-label="Start Task"
    onclick={() => {
      taskStarted = true;
      manager.start();
    }}
    disabled={taskStarted}
  >
    {taskStarted ? "Task Started" : "Start Task"}
  </button>
</div>
