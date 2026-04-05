import coreCards from '../eval/local_intelligence/v2/core_eval_v2_cards.json' with { type: 'json' }
import v2AnchorRun from '../eval/local_intelligence/results/github_models-20260405T175944Z.json' with { type: 'json' }

export function getCoreEvalCardsPreview() {
  const preferred = [
    'memo-auto-title',
    'memo-type-detection',
    'transcript-cleanup-presets',
    'private-redaction-pass',
    'action-item-extraction',
    'reminder-normalization',
    'calendar-intent-detection',
    'follow-up-question-generator',
  ]

  const byId = new Map(coreCards.map((card) => [card.id, card]))
  return preferred.map((id) => byId.get(id)).filter(Boolean)
}

export function getV2AnchorSummary() {
  return v2AnchorRun.summary
}

export function getV1ScoreRows() {
  return [
    {
      model: 'google/gemma-4-E4B-it',
      pass: '13/24',
      avg: '0.62',
      note: 'Best Gemma family result so far',
    },
    {
      model: 'google/gemma-4-E2B-it',
      pass: '8/24',
      avg: '0.49',
      note: 'Smaller open baseline with visible drop',
    },
    {
      model: 'nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1',
      pass: '2/24',
      avg: '0.19',
      note: 'Useful weak-end contrast',
    },
  ]
}
