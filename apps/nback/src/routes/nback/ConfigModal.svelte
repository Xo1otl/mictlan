<script lang="ts">
    import MultiSelect from "svelte-multiselect";
    import * as nback from "../../nback/index";
    import type { Config } from "./+page";

    let {
        showModal = $bindable(),
        config = $bindable(),
    }: {
        showModal: boolean;
        config: Config;
    } = $props();

    // 初期設定の読み込み、
    // 値はmodalが開く時に初期化されるのでderiveでなくても問題ないが多分書き直したほうがいい
    const { trialFactoryOptions, taskEngineOptions } = config;

    /* --- Task Engine 用設定 --- */
    let engineN = $state(taskEngineOptions.n);
    let problemCount = $state(taskEngineOptions.problemCount);
    let interval = $state(taskEngineOptions.interval);

    /* --- その他の設定 --- */
    let answerDisplayTime = $state(config.answerDisplayTime);

    /* --- Trial Factory 用設定 --- */
    // 刺激種別チェックボックスの初期状態
    const stimulusTypes = trialFactoryOptions?.stimulusTypes || [];

    let stimulusCharacter = $state(
        stimulusTypes.includes(nback.StimulusType.Character),
    );
    let stimulusPosition = $state(
        stimulusTypes.includes(nback.StimulusType.Position),
    );
    let stimulusColor = $state(
        stimulusTypes.includes(nback.StimulusType.Color),
    );
    let stimulusAnimation = $state(
        stimulusTypes.includes(nback.StimulusType.Animation),
    );
    let stimulusShape = $state(
        stimulusTypes.includes(nback.StimulusType.Shape),
    );
    let stimulusSound = $state(
        stimulusTypes.includes(nback.StimulusType.Sound),
    );

    // 各 enum 用の MultiSelect の選択肢と初期選択状態（初期設定がある場合はそれを使い、なければ全選択をデフォルトとする）
    let availableCharacters = Object.values(nback.Character);
    let selectedCharacters = $state(
        trialFactoryOptions?.characters || availableCharacters,
    );

    let availableColors = Object.values(nback.Color);
    let selectedColors = $state(trialFactoryOptions?.colors || availableColors);

    let availableAnimations = Object.values(nback.Animation);
    let selectedAnimations = $state(
        trialFactoryOptions?.animations || availableAnimations,
    );

    let availableShapes = Object.values(nback.Shape);
    let selectedShapes = $state(trialFactoryOptions?.shapes || availableShapes);

    let availableSounds = Object.values(nback.Sound);
    let selectedSounds = $state(trialFactoryOptions?.sounds || availableSounds);

    // gridSize の初期値（Position 刺激用）
    let gridWidth = $state(
        trialFactoryOptions?.gridSize
            ? trialFactoryOptions.gridSize[0]
            : nback.DefaultTrialFactoryOptions.gridSize[0],
    );
    let gridHeight = $state(
        trialFactoryOptions?.gridSize
            ? trialFactoryOptions.gridSize[1]
            : nback.DefaultTrialFactoryOptions.gridSize[1],
    );

    // これが呼ばれるまでconfigの値は更新されない
    const applyOptions = (
        event: SubmitEvent & {
            currentTarget: EventTarget & HTMLFormElement;
        },
    ) => {
        event.preventDefault();

        const newTrialFactoryOptions: nback.TrialFactoryOptions & {
            stimulusTypes: nback.StimulusType[];
        } = {
            stimulusTypes: [],
        };

        // 各刺激タイプごとにチェック状態に応じた設定を追加
        if (stimulusCharacter) {
            newTrialFactoryOptions.stimulusTypes.push(
                nback.StimulusType.Character,
            );
            newTrialFactoryOptions.characters = selectedCharacters;
        }
        if (stimulusColor) {
            newTrialFactoryOptions.stimulusTypes.push(nback.StimulusType.Color);
            newTrialFactoryOptions.colors = selectedColors;
        }
        if (stimulusAnimation) {
            newTrialFactoryOptions.stimulusTypes.push(
                nback.StimulusType.Animation,
            );
            newTrialFactoryOptions.animations = selectedAnimations;
        }
        if (stimulusShape) {
            newTrialFactoryOptions.stimulusTypes.push(nback.StimulusType.Shape);
            newTrialFactoryOptions.shapes = selectedShapes;
        }
        if (stimulusSound) {
            newTrialFactoryOptions.stimulusTypes.push(nback.StimulusType.Sound);
            newTrialFactoryOptions.sounds = selectedSounds;
        }
        if (stimulusPosition) {
            newTrialFactoryOptions.stimulusTypes.push(
                nback.StimulusType.Position,
            );
            newTrialFactoryOptions.gridSize = [gridWidth, gridHeight];
        }

        const newTaskEngineOptions = {
            n: engineN,
            problemCount: problemCount,
            interval: interval,
        };

        config = {
            trialFactoryOptions: newTrialFactoryOptions,
            taskEngineOptions: newTaskEngineOptions,
            answerDisplayTime: answerDisplayTime,
        };

        showModal = false;
    };
</script>

<!-- モーダル全体 -->
<div class="fixed inset-0 flex items-center justify-center z-50 p-4">
    <!-- バックドロップ -->
    <button
        type="button"
        onclick={() => (showModal = false)}
        aria-label="Close modal"
        class="absolute inset-0 bg-black bg-opacity-50"
    ></button>

    <!-- モーダルコンテンツ -->
    <div
        class="relative bg-white p-6 rounded-lg shadow-lg z-10 w-full max-w-2xl mx-auto max-h-[90vh] overflow-y-auto"
    >
        <h2 class="text-2xl font-bold mb-4">設定を変更</h2>
        <form onsubmit={applyOptions} class="space-y-6">
            <!-- Task Engine 設定 -->
            <fieldset class="border border-gray-200 p-4 rounded">
                <legend class="font-bold mb-2">Task Engine 設定</legend>
                <div class="grid grid-cols-1 gap-4">
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700"
                            >N:</span
                        >
                        <input
                            type="number"
                            bind:value={engineN}
                            min="1"
                            required
                            class="mt-1 block w-full border border-gray-300 rounded p-2 focus:ring-blue-500 focus:border-blue-500"
                        />
                    </label>
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            問題数:
                        </span>
                        <input
                            type="number"
                            bind:value={problemCount}
                            min="1"
                            required
                            class="mt-1 block w-full border border-gray-300 rounded p-2 focus:ring-blue-500 focus:border-blue-500"
                        />
                    </label>
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            問題表示間隔 (ms):
                        </span>
                        <input
                            type="number"
                            bind:value={interval}
                            min="3000"
                            required
                            class="mt-1 block w-full border border-gray-300 rounded p-2 focus:ring-blue-500 focus:border-blue-500"
                        />
                    </label>
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            解答表示時間 (ms):
                        </span>
                        <input
                            type="number"
                            bind:value={answerDisplayTime}
                            min="0"
                            max={interval}
                            required
                            class="mt-1 block w-full border border-gray-300 rounded p-2 focus:ring-blue-500 focus:border-blue-500"
                        />
                    </label>
                </div>
            </fieldset>

            <!-- Trial Factory 設定 -->
            <fieldset class="border border-gray-200 p-4 rounded">
                <legend class="font-bold mb-2">Trial Factory 設定</legend>
                <!-- 刺激タイプ（チェックボックス） -->
                <div class="mb-4">
                    <span class="block text-sm font-medium text-gray-700 mb-1">
                        刺激タイプ:
                    </span>
                    <div class="flex flex-wrap gap-4">
                        <label class="inline-flex items-center">
                            <input
                                type="checkbox"
                                bind:checked={stimulusCharacter}
                                class="mr-2"
                            />
                            文字
                        </label>
                        <label class="inline-flex items-center">
                            <input
                                type="checkbox"
                                bind:checked={stimulusPosition}
                                class="mr-2"
                            />
                            位置
                        </label>
                        <label class="inline-flex items-center">
                            <input
                                type="checkbox"
                                bind:checked={stimulusColor}
                                class="mr-2"
                            />
                            色
                        </label>
                        <label class="inline-flex items-center">
                            <input
                                type="checkbox"
                                bind:checked={stimulusAnimation}
                                class="mr-2"
                            />
                            アニメーション
                        </label>
                        <label class="inline-flex items-center">
                            <input
                                type="checkbox"
                                bind:checked={stimulusShape}
                                class="mr-2"
                            />
                            形
                        </label>
                        <label class="inline-flex items-center">
                            <input
                                type="checkbox"
                                bind:checked={stimulusSound}
                                class="mr-2"
                            />
                            音
                        </label>
                    </div>
                </div>

                <!-- 各 enum 用 MultiSelect コンポーネント -->
                {#if stimulusCharacter}
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            Characters:
                        </span>
                        <MultiSelect
                            bind:selected={selectedCharacters}
                            options={availableCharacters}
                        />
                    </label>
                {/if}

                {#if stimulusColor}
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            Colors:
                        </span>
                        <MultiSelect
                            bind:selected={selectedColors}
                            options={availableColors}
                        />
                    </label>
                {/if}

                {#if stimulusAnimation}
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            Animations:
                        </span>
                        <MultiSelect
                            bind:selected={selectedAnimations}
                            options={availableAnimations}
                        />
                    </label>
                {/if}

                {#if stimulusShape}
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            Shapes:
                        </span>
                        <MultiSelect
                            bind:selected={selectedShapes}
                            options={availableShapes}
                        />
                    </label>
                {/if}

                {#if stimulusSound}
                    <label class="block">
                        <span class="block text-sm font-medium text-gray-700">
                            Sounds:
                        </span>
                        <MultiSelect
                            bind:selected={selectedSounds}
                            options={availableSounds}
                        />
                    </label>
                {/if}

                {#if stimulusPosition}
                    <!-- Position の場合は gridSize の設定 -->
                    <div class="grid grid-cols-2 gap-4">
                        <label class="block">
                            <span
                                class="block text-sm font-medium text-gray-700"
                            >
                                Grid Width:
                            </span>
                            <input
                                type="number"
                                bind:value={gridWidth}
                                min="1"
                                required
                                class="mt-1 block w-full border border-gray-300 rounded p-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                        </label>
                        <label class="block">
                            <span
                                class="block text-sm font-medium text-gray-700"
                            >
                                Grid Height:
                            </span>
                            <input
                                type="number"
                                bind:value={gridHeight}
                                min="1"
                                required
                                class="mt-1 block w-full border border-gray-300 rounded p-2 focus:ring-blue-500 focus:border-blue-500"
                            />
                        </label>
                    </div>
                {/if}
            </fieldset>

            <!-- ボタン -->
            <div class="flex justify-end gap-4 pt-4 border-t border-gray-200">
                <button
                    type="button"
                    onclick={() => (showModal = false)}
                    class="px-4 py-2 border rounded text-gray-700 hover:bg-gray-100 transition-colors"
                >
                    キャンセル
                </button>
                <button
                    type="submit"
                    class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                >
                    適用
                </button>
            </div>
        </form>
    </div>
</div>
