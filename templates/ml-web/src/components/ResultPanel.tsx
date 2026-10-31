import { ArrowDownRight, ArrowUpRight, LoaderCircle, Minus, TriangleAlert } from "lucide-react";

import type { Feature, WhatIfOut } from "@/api/client";
import { Alert, Badge, Skeleton } from "@/components/ui/feedback";
import { pct, points } from "@/lib/format";

interface Props {
  result: WhatIfOut | undefined;
  pending: boolean;
  features: Feature[];
  targetLabel: string;
}

export function ResultPanel({ result, pending, features, targetLabel }: Props) {
  if (!result) {
    return (
      <section aria-label="Prediction" aria-busy="true" className="flex flex-col gap-3">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-12 w-32" />
        <Skeleton className="h-4 w-64" />
      </section>
    );
  }
  const { scenario, baseline, delta } = result;
  const above = scenario.label === "above threshold";
  const outside = Object.entries(scenario.extrapolation).filter(([, v]) => v).map(([k]) => features.find((f) => f.name === k)?.label ?? k);
  const DeltaIcon = delta > 0.0005 ? ArrowUpRight : delta < -0.0005 ? ArrowDownRight : Minus;
  return (
    <section aria-label="Prediction" aria-busy={pending} className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Predicted probability of {targetLabel}</h2>
        {pending ? (
          <span className="flex items-center gap-1 text-xs text-muted-foreground" role="status">
            <LoaderCircle aria-hidden className="size-3.5 animate-spin motion-reduce:animate-none" /> Updating
          </span>
        ) : null}
      </div>
      <div className="flex flex-wrap items-end gap-3" aria-live="polite">
        <p className="tabular text-4xl font-semibold leading-tight">{pct(scenario.probability)}</p>
        <Badge variant={above ? "warning" : "neutral"}>
          {above ? <TriangleAlert aria-hidden /> : null}
          {above ? "Flagged" : "Not flagged"} at threshold {pct(scenario.threshold, 0)}
        </Badge>
      </div>
      <p className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        Median profile {pct(baseline.probability)}
        <Badge variant="neutral" className="tabular">
          <DeltaIcon aria-hidden />
          {points(delta)}
        </Badge>
      </p>
      {outside.length ? (
        <Alert variant="warning">
          <TriangleAlert aria-hidden />
          <p>
            {outside.join(", ")} {outside.length === 1 ? "is" : "are"} outside the training range. The model has not seen values like
            this, so treat the prediction as an extrapolation.
          </p>
        </Alert>
      ) : null}
    </section>
  );
}
