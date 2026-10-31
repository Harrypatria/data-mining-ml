import type { ModelInfo, SchemaOut, WhatIfOut } from "@/api/client";

export const schema: SchemaOut = {
  columns: ["tenure_months", "plan", "autopay"],
  features: [
    { name: "tenure_months", label: "Tenure", unit: "months", description: "Months since first invoice.", type: "integer", adjustable: true, min: 1, max: 72, step: 1, default: 37 },
    { name: "plan", label: "Plan", unit: "", description: "Subscription tier.", type: "category", adjustable: true, default: "basic", allowed: ["basic", "premium", "standard"] },
    { name: "autopay", label: "Autopay", unit: "", description: "Paid automatically.", type: "boolean", adjustable: true, default: true },
  ],
};

export const model: ModelInfo = {
  model_version: "1.0.0",
  created_utc: "2026-01-01T00:00:00+00:00",
  task: "binary_classification",
  target: "churned",
  target_label: "cancelling within 30 days",
  model_family: "logistic",
  threshold: 0.22,
  threshold_rule: "max recall with precision >= 0.35",
  metrics: { pr_auc: { value: 0.5, low: 0.45, high: 0.55 } },
  test_positive_count: 180,
  limitations: ["Simulated data."],
  calibration_note: "Check calibration.",
  tested_model_note: null,
  model_purpose: "Flagging customers likely to cancel",
};

export function whatIf(probability: number): WhatIfOut {
  const record = (p: number) => ({
    probability: p, value: null, label: p >= 0.22 ? "above threshold" : "below threshold", threshold: 0.22,
    model_version: "1.0.0", request_id: "r1", extrapolation: { tenure_months: false, plan: false, autopay: false }, extrapolated: false,
  });
  return {
    baseline: record(0.1),
    scenario: record(probability),
    delta: probability - 0.1,
    changed: [],
    base_value: 0.12,
    contributions: {
      baseline: [],
      scenario: [
        { feature: "tenure_months", value: 37, contribution: probability - 0.12, direction: "raises" },
        { feature: "plan", value: "basic", contribution: 0, direction: "no effect" },
      ],
    },
  };
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

export function sseResponse(events: Array<[string, unknown]>): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const [event, data] of events) controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
}
