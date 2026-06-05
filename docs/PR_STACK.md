# RoboClaw PR Stack

This workspace currently contains multiple product-level changes. Do not submit
them as one pull request. Split them into the PR stack below so each review has a
clear user value, risk boundary, and verification path.

## PR 1: Show Cloud Training Artifacts And Metrics

Goal: after a cloud task finishes, the training page must show metrics and
artifact paths instead of hiding them in logs.

Primary files:

- `roboclaw/http/routes/train_cloud.py`
- `roboclaw/http/routes/cloud_ssh.py`
- `ui/src/domains/training/components/TrainingProgressPanel.tsx`
- `ui/src/domains/training/store/useTrainingStore.ts`
- `tests/test_evo_train_routes.py`

Expected behavior:

- `/api/train/cloud/current` and `/api/train/cloud/status/{job_id}` include
  `artifacts`, `metricsPath`, and metrics parsed from cloud logs.
- `/api/train/cloud/artifacts` exposes previewable artifact paths and attempts to
  read metrics JSON when SSH binding is available.
- The training page keeps a completed cloud job visible long enough to show
  metrics and artifact paths.
- Artifact paths are visible by default.

Verification:

```bash
python -m pytest tests/test_evo_train_routes.py -k artifacts -q
cd ui && npm run build
```

## PR 2: Stabilize Cloud Job State And Repair Supervisor

Goal: make cloud jobs continue across web restarts and stop repeatedly creating
confusing repair task names.

Primary files:

- `roboclaw/http/routes/cloud_supervisor.py`
- `roboclaw/http/routes/cloud_autonomy.py`
- `roboclaw/http/routes/cloud_repair_agent.py`
- `roboclaw/http/routes/train_cloud.py`
- `roboclaw/http/routes/train_cloud_helpers.py`
- `roboclaw/http/routes/train_cloud_schema.py`
- `roboclaw/cloud/evo_train.py`
- `roboclaw/training/service.py`
- `tests/test_evo_train_routes.py`

Expected behavior:

- Active repair/intervention jobs are watched again after the web server
  restarts.
- Terminal success clears stale failure state and releases frozen billing holds.
- Repair jobs preserve enough context to avoid repeating identical failed
  commands.
- Training-time intervention can classify common runtime failures such as Ray
  GCS startup failures.

Verification:

```bash
python -m pytest tests/test_evo_train_routes.py -q
```

## PR 3: Simplify Training Center UX

Goal: make cloud training feel like a single AI task flow rather than a large
form.

Primary files:

- `ui/src/domains/training/pages/TrainingCenterPage.tsx`
- `ui/src/domains/training/components/CloudIntentPanel.tsx`
- `ui/src/domains/training/components/CloudProviderPanel.tsx`
- `ui/src/domains/training/components/CloudSourcePanel.tsx`
- `ui/src/domains/training/components/TrainingProgressPanel.tsx`
- `ui/src/domains/training/store/useTrainingStore.ts`

Expected behavior:

- One primary action for cloud task submission.
- Current plan, current task, and history are visually separated.
- Debug/provider details are not in the main user path.
- Failure handling appears as a small resumable window instead of blocking the
  whole page.
- Finished tasks do not disappear before users can inspect results.

Verification:

```bash
cd ui && npm run build
```

Manual browser checks:

- Open `/training`.
- Confirm a completed cloud task shows results and paths.
- Confirm failure dialog can collapse and reopen.
- Confirm repeated refresh does not flash or clear the completed task.

## PR 4: Dataset Push And Data Asset Flow

Goal: build the first step of the robot data flywheel: one command or simple UI
flow to upload/register datasets and receive credit.

Primary files:

- `roboclaw/cli/datasets.py`
- `roboclaw/data/dataset_push.py`
- `roboclaw/data/storage.py`
- `roboclaw/data/auth_refs.py`
- `roboclaw/http/routes/datasets.py`
- `roboclaw/http/routes/hub.py`
- `ui/src/domains/datasets/pages/DatasetsPage.tsx`
- `ui/src/domains/datasets/store/useDatasetsStore.ts`
- `ui/src/domains/datasets/types.ts`
- `tests/test_dataset_push_cli.py`
- `tests/test_dataset_upload_completion.py`

Expected behavior:

- `roboclaw dataset push ./dataset` can package/register a dataset.
- Dataset UI no longer silently defaults new users to `pearl`.
- Push-to-Hub uses inline inputs instead of `window.prompt`.
- Status messages are accessible via `aria-live`.

Verification:

```bash
python -m pytest tests/test_dataset_push_cli.py tests/test_dataset_upload_completion.py -q
cd ui && npm run build
```

## PR 5: AI Provider And Tool-Calling Robustness

Goal: make provider configuration and fallback behavior explicit enough for
research users using custom/relay APIs.

Primary files:

- `roboclaw/providers/base.py`
- `roboclaw/providers/custom_provider.py`
- `roboclaw/providers/factory.py`
- `roboclaw/providers/litellm_provider.py`
- `roboclaw/providers/openai_codex_provider.py`
- `ui/src/domains/provider/api/providerApi.ts`
- `ui/src/domains/settings/pages/ProviderSettingsPage.tsx`
- `tests/test_provider_text_tool_fallback.py`
- `tests/test_web_provider_api.py`

Expected behavior:

- Provider errors surface as real errors instead of fake assistant messages.
- Text-only fallback is explicit when a relay does not support tools.
- Settings page explains model/provider choices without hiding invalid tokens,
  insufficient balance, or malformed tool payloads.

Verification:

```bash
python -m pytest tests/test_provider_text_tool_fallback.py tests/test_web_provider_api.py -q
cd ui && npm run build
```

## PR 6: Codebase Hygiene And Route Splitting

Goal: reduce large files, remove silent error swallowing, and make operational
failures visible.

Primary files:

- `roboclaw/cli/commands.py`
- `roboclaw/cli/interactive.py`
- `roboclaw/cli/provider_auth.py`
- `roboclaw/data/curation/service.py`
- `roboclaw/data/curation/scoring.py`
- `roboclaw/data/curation/propagation.py`
- `roboclaw/embodied/embodiment/hardware/*`
- `roboclaw/embodied/toolkit/tools.py`
- `roboclaw/embodied/executor.py`
- `roboclaw/agent/loop.py`
- `roboclaw/channels/feishu.py`
- `ui/src/shared/api/client.ts`
- `ui/src/i18n/store.ts`

Expected behavior:

- Large files move toward the single-file-under-1000-lines goal.
- Probe/storage/provider failures log warnings instead of disappearing.
- Tool errors raise through the normal error path where appropriate.

Verification:

```bash
python -m pytest tests/ -x -q
cd ui && npm run build
```

## Submission Order

1. PR 1 first because it is the smallest user-visible bug fix.
2. PR 2 next because it changes cloud job semantics and billing release logic.
3. PR 3 after PR 2, because the UI depends on stable state semantics.
4. PR 4 can run in parallel if dataset reviewers are separate.
5. PR 5 can run in parallel if provider reviewers are separate.
6. PR 6 last because it is broad and mostly structural.

