import { FileText, Moon, Sun } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, type Profile, type ProfileValue } from "@/api/client";
import { Charts } from "@/components/ChartsSection";
import { ExplainerChat } from "@/components/ExplainerChat";
import { ModelDrawer, ScenarioTable, StatusPanel, type SavedScenario } from "@/components/Panels";
import { PredictorPanel } from "@/components/PredictorPanel";
import { ResultPanel } from "@/components/ResultPanel";
import { Button } from "@/components/ui/button";
import { Alert, Skeleton } from "@/components/ui/feedback";
import { Sheet, SheetTrigger } from "@/components/ui/sheet";
import { TooltipProvider } from "@/components/ui/tooltip";
import { medianProfile, useExplainer, useModelInfo, useSchema, useWhatIf } from "@/hooks/useWorkbench";
import { applyTheme, currentTheme } from "@/lib/theme";

export default function App() {
  const schema = useSchema();
  const model = useModelInfo(schema.isSuccess);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [allowExtrapolation, setAllowExtrapolation] = useState(false);
  const [lastMoved, setLastMoved] = useState<string | undefined>();
  const [scenarios, setScenarios] = useState<SavedScenario[]>([]);
  const saved = useRef(0);
  const [autoExplain, setAutoExplain] = useState(false);
  const [theme, setTheme] = useState(currentTheme);
  const explainer = useExplainer();

  useEffect(() => {
    if (schema.data && profile === null) setProfile(medianProfile(schema.data));
  }, [schema.data, profile]);

  const whatIf = useWhatIf(profile, allowExtrapolation);
  const features = useMemo(() => schema.data?.features ?? [], [schema.data]);
  const moved = features.find((f) => f.name === lastMoved) ?? features.find((f) => f.type === "number" || f.type === "integer");

  const onChange = useCallback((name: string, value: ProfileValue) => {
    setProfile((p) => (p ? { ...p, [name]: value } : p));
    setLastMoved(name);
  }, []);

  // Auto-explain fires once per settled scenario, never while a slider is moving (the what-if result is debounced).
  const explained = useRef<string>("");
  useEffect(() => {
    if (!autoExplain || !profile || !whatIf.data || whatIf.isFetching) return;
    const key = JSON.stringify(profile);
    if (key === explained.current || explainer.status === "streaming") return;
    explained.current = key;
    explainer.ask("Explain this change", profile, allowExtrapolation);
  }, [autoExplain, profile, whatIf.data, whatIf.isFetching, explainer, allowExtrapolation]);

  const fatal = schema.error instanceof ApiError ? schema.error : null;
  const whatIfError = whatIf.error instanceof ApiError ? whatIf.error : null;
  const targetLabel = model.data?.target_label ?? "the outcome";

  return (
    <TooltipProvider delayDuration={200}>
      <a href="#workbench" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2">
        Skip to workbench
      </a>
      <header className="flex items-center justify-between gap-4 border-b border-border bg-card px-4 py-3 lg:px-6">
        <div className="flex flex-col">
          <h1 className="text-lg font-semibold leading-tight">Prediction workbench</h1>
          <p className="text-xs text-muted-foreground">
            {model.data ? `${model.data.model_purpose} - model v${model.data.model_version}` : "Loading model details"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="outline" size="sm">
                <FileText aria-hidden /> Model information
              </Button>
            </SheetTrigger>
            <ModelDrawer info={model.data} />
          </Sheet>
          <Button
            variant="ghost"
            size="icon"
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            onClick={() => setTheme(applyTheme(theme === "dark" ? "light" : "dark", true))}
          >
            {theme === "dark" ? <Sun aria-hidden /> : <Moon aria-hidden />}
          </Button>
        </div>
      </header>

      <main id="workbench" className="mx-auto w-full max-w-screen-2xl p-4 lg:p-6">
        {fatal ? (
          <StatusPanel error={fatal} onRetry={() => schema.refetch()} />
        ) : !schema.data || !profile ? (
          <div className="grid gap-6 lg:grid-cols-3" aria-busy="true" aria-label="Loading the workbench">
            <Skeleton className="h-96" />
            <Skeleton className="h-96" />
            <Skeleton className="h-96" />
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-12">
            <aside className="rounded-lg border border-border bg-card p-4 lg:col-span-4 xl:col-span-3">
              <PredictorPanel
                features={features}
                profile={profile}
                allowExtrapolation={allowExtrapolation}
                onChange={onChange}
                onAllowExtrapolation={setAllowExtrapolation}
                onReset={() => schema.data && setProfile(medianProfile(schema.data))}
                onSave={() => {
                  saved.current += 1;
                  const name = `Scenario ${saved.current}`;
                  setScenarios((s) => [...s, { name, profile, probability: whatIf.data?.scenario.probability ?? null }]);
                }}
              />
            </aside>

            <div className="flex flex-col gap-6 lg:col-span-8 xl:col-span-6">
              <div className="rounded-lg border border-border bg-card p-4">
                {whatIfError ? (
                  <Alert variant="danger">
                    <span className="sr-only">Error:</span>
                    <p>{whatIfError.message}</p>
                  </Alert>
                ) : (
                  <ResultPanel result={whatIf.data} pending={whatIf.isFetching} features={features} targetLabel={targetLabel} />
                )}
              </div>
              <div className="rounded-lg border border-border bg-card p-4">
                <Charts whatIf={whatIf.data} features={features} moved={moved} profile={profile} allowExtrapolation={allowExtrapolation} />
              </div>
              <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
                <h2 className="text-sm font-semibold">Scenario comparison</h2>
                <ScenarioTable scenarios={scenarios} features={features} onRemove={(name) => setScenarios((s) => s.filter((x) => x.name !== name))} />
              </div>
            </div>

            <aside className="rounded-lg border border-border bg-card p-4 lg:col-span-12 xl:col-span-3">
              <ExplainerChat
                messages={explainer.messages}
                status={explainer.status}
                error={explainer.error}
                autoExplain={autoExplain}
                onAutoExplain={setAutoExplain}
                onAsk={(q) => explainer.ask(q, profile, allowExtrapolation)}
              />
            </aside>
          </div>
        )}
      </main>
    </TooltipProvider>
  );
}
