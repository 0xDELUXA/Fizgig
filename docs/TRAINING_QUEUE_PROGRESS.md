# Training Queue — branch progress (queue-tests)

Status: **built and headless-tested, awaiting real-session validation.** One commit:
`c1e565d` (GUI). The test suite (`tests/test_training_queue.py`, 20 asserts, all passing)
lives untracked on disk per the tests-off-GitHub policy.

## What's built

- **Queue Train**: while a run is active, the Start Training button relabels to
  "Queue Train" and appends the currently configured run to the queue instead of
  popping "Already Running". (`start_training` re-entrancy guard + `_refresh_training_buttons`.)
- **Queue item** = `_collect_preset_values()` snapshot **plus** the three things presets
  deliberately omit: architecture, Start-tab image folder, Samples-tab entries.
  Restore = load into GUI → normal `start_training()`; the queue never bypasses
  validation/TOML/caching/pause machinery.
- **Status-bar button**, lower-right corner: `📋 Queue (N)`, accent-coloured when non-empty.
  Sample-override widths trimmed (seed 8→6, W/H combos 6→5, paddings) to make room.
- **Queue window** (`_open_queue_window`): card per run — thumbnail (first dataset image),
  name, summary (arch, folder + image count, LR, epochs, dim, type, area, queued-at) —
  with ↑ ↓ reorder, ✎ load-into-tab, ⤓ update-from-tab, ✕ delete; footer: Start next now,
  Clear queue.
- **Pickup**: `_on_training_subprocess_exited` — clean exit + state "running" + queue
  non-empty → next run auto-starts after 5 s. **Pod auto-stop defers until the queue
  drains** (last clean exit fires it).
- **Hold policy**: failure / user Stop / Pause freeze the queue (no overnight crash
  cascades); console explains; restart from the queue window. A queued run that fails
  validation at its turn returns to the head of the queue.
- **Persistence**: `presets/training_queue.json`, atomic writes, `FIZGIG_NO_PERSIST`
  guard. Survives restart; **never auto-starts on launch**.

## Method map (all in `lora_trainer_gui.py`)

`QUEUE_FILE` constant · init hook loading `self.training_queue` · block of `_queue_*` /
`_load_training_queue` / `_save_training_queue` methods just above `_NON_TRAINING_ENTRY_KEYS`
· router inside `start_training` · button flip in `_refresh_training_buttons` · pickup in
`_on_training_subprocess_exited` · status-bar button in the status-bar builder.

## Known v1 simplifications / remaining work

1. **Sample prompts** come from the Samples tab as-is at each run's launch; only the
   SAMPLE_* settings entries are captured per item. Capturing the prompt list per run
   means snapshotting the Samples tab's prompt store — decide if wanted.
2. **Real-session validation**: queue two short runs, verify hand-off, then a failure
   mid-queue (hold), a Stop (hold), pod auto-stop deferral, and restart persistence.
3. Edit flow is load-into-tab + write-back (⤓) rather than an in-window editor — by
   design (the Training tab IS the editor); revisit only if it confuses in practice.
4. Queue button lives in the collapsible status bar — if users hide the bar the button
   hides too. Option: mirror a count on the always-visible handle strip.
5. Cross-arch queues switch the Base Model selector between runs (handled via
   `update_ui_for_architecture`) — worth one real Klein→Krea 2 queue test.
6. On merge: README section + release notes ("queue up overnight runs"), and a pod
   angle — queue + auto-stop = fire-and-forget rented-GPU batches.
