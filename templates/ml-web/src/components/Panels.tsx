import { CircleAlert, CloudOff, Hourglass, Lock, RefreshCw, Trash2 } from "lucide-react";

import type { ApiError, Feature, ModelInfo, Profile } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/feedback";
import { SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { formatValue, pct } from "@/lib/format";

export interface SavedScenario {
  name: string;
  profile: Profile;
  probability: number | null;
}

export function ScenarioTable({ scenarios, features, onRemove }: { scenarios: SavedScenario[]; features: Feature[]; onRemove: (name: string) => void }) {
  if (scenarios.length === 0) {
    return (
      <p className="rounded-md border border-dashed border-border p-3 text-sm text-muted-foreground">
        No saved scenarios yet. Use Save scenario to keep the current profile and compare it here.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <caption className="sr-only">Saved scenarios with their predicted probability</caption>
        <thead className="text-xs text-muted-foreground">
          <tr>
            <th scope="col" className="py-2 pr-3 font-medium">Scenario</th>
            <th scope="col" className="py-2 pr-3 font-medium">Probability</th>
            {features.map((f) => (
              <th key={f.name} scope="col" className="py-2 pr-3 font-medium">{f.label}</th>
            ))}
            <th scope="col" className="py-2"><span className="sr-only">Remove</span></th>
          </tr>
        </thead>
        <tbody>
          {scenarios.map((s) => (
            <tr key={s.name} className="border-t border-border">
              <th scope="row" className="py-2 pr-3 font-medium">{s.name}</th>
              <td className="tabular py-2 pr-3">{pct(s.probability)}</td>
              {features.map((f) => (
                <td key={f.name} className="tabular py-2 pr-3 text-muted-foreground">{formatValue(s.profile[f.name], f)}</td>
              ))}
              <td className="py-2">
                <Button variant="ghost" size="icon" aria-label={`Remove ${s.name}`} onClick={() => onRemove(s.name)}>
                  <Trash2 aria-hidden />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelDrawer({ info }: { info: ModelInfo | undefined }) {
  return (
    <SheetContent aria-describedby="model-drawer-description">
      <SheetTitle className="text-lg font-semibold">Model information</SheetTitle>
      <SheetDescription id="model-drawer-description" className="text-sm text-muted-foreground">
        What this model is for, how well it did on held-out data, and where it should not be trusted.
      </SheetDescription>
      {!info ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-24 w-full" />
        </div>
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground">Version</dt>
            <dd className="tabular">{info.model_version}</dd>
            <dt className="text-muted-foreground">Trained</dt>
            <dd className="tabular">{new Date(info.created_utc).toLocaleDateString()}</dd>
            <dt className="text-muted-foreground">Family</dt>
            <dd>{info.model_family?.replace(/_/g, " ") ?? "n/a"}</dd>
            <dt className="text-muted-foreground">Decision threshold</dt>
            <dd className="tabular">{pct(info.threshold, 0)}</dd>
          </dl>
          <section className="flex flex-col gap-2">
            <h3 className="text-sm font-semibold">Intended use</h3>
            <p className="text-sm">{info.model_purpose}</p>
          </section>
          <section className="flex flex-col gap-2">
            <h3 className="text-sm font-semibold">Test results with 95% intervals</h3>
            <table className="text-sm">
              <caption className="sr-only">Metrics on the held-out test set</caption>
              <tbody>
                {Object.entries(info.metrics).map(([name, m]) => (
                  <tr key={name} className="border-t border-border">
                    <th scope="row" className="py-1.5 pr-3 text-left font-medium uppercase">{name.replace(/_/g, " ")}</th>
                    <td className="tabular py-1.5 text-right">
                      {m.value?.toFixed(3)} <span className="text-muted-foreground">({m.low?.toFixed(3)} to {m.high?.toFixed(3)})</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {info.test_positive_count != null ? (
              <p className="text-xs text-muted-foreground">Based on {info.test_positive_count} positive cases in the test set.</p>
            ) : null}
            {info.tested_model_note ? <p className="text-xs text-muted-foreground">{info.tested_model_note}</p> : null}
          </section>
          <section className="flex flex-col gap-2">
            <h3 className="text-sm font-semibold">Limitations</h3>
            <ul className="list-disc pl-5 text-sm">
              {info.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
            <p className="text-xs text-muted-foreground">{info.calibration_note}</p>
          </section>
        </>
      )}
    </SheetContent>
  );
}

const STATUS = {
  unavailable: { icon: CloudOff, title: "Service unavailable" },
  not_ready: { icon: Hourglass, title: "Model not ready" },
  rate_limited: { icon: Hourglass, title: "Too many requests" },
  unauthorized: { icon: Lock, title: "Access refused" },
  invalid: { icon: CircleAlert, title: "Inputs rejected" },
  server: { icon: CircleAlert, title: "Something went wrong" },
} as const;

export function StatusPanel({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  const { icon: Icon, title } = STATUS[error.kind];
  return (
    <div role="alert" className="mx-auto flex max-w-md flex-col items-center gap-3 rounded-lg border border-border bg-card p-8 text-center">
      <Icon aria-hidden className="size-6 text-muted-foreground" />
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="text-sm text-muted-foreground">{error.message}</p>
      <Button variant="outline" onClick={onRetry}>
        <RefreshCw aria-hidden /> Try again
      </Button>
    </div>
  );
}
