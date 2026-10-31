import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from "recharts";

import type { Contribution, Feature } from "@/api/client";
import { Skeleton } from "@/components/ui/feedback";
import { formatValue, pct, points } from "@/lib/format";

const AXIS = { fontSize: 12, fill: "var(--color-text-muted)" };

export function ContributionChart({ contributions, features }: { contributions: Contribution[] | undefined; features: Feature[] }) {
  if (!contributions) return <Skeleton className="h-56 w-full" />;
  const label = (name: string) => features.find((f) => f.name === name)?.label ?? name;
  const rows = contributions.map((c) => ({ ...c, name: `${label(c.feature)}  ${points(c.contribution)}`, text: points(c.contribution) }));
  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="text-sm font-semibold">What pushes this prediction up or down</figcaption>
      <div className="h-56 w-full" aria-hidden>
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
            <CartesianGrid horizontal={false} stroke="var(--color-data-grid)" />
            <XAxis type="number" tick={AXIS} tickFormatter={(v: number) => points(v, 0)} />
            <YAxis type="category" dataKey="name" width={170} tick={AXIS} />
            <ReferenceLine x={0} stroke="var(--color-border-strong)" />
            <Bar dataKey="contribution" isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.feature} fill={r.contribution >= 0 ? "var(--color-data-raises)" : "var(--color-data-lowers)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="sr-only">
        <caption>Contribution of each predictor, in percentage points</caption>
        <tbody>
          {rows.map((r) => (
            <tr key={r.feature}>
              <th scope="row">{label(r.feature)}</th>
              <td>{r.text}</td>
              <td>{r.direction}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-muted-foreground">
        Contributions describe what the model learned, not causes. Predictors that move together share credit.
      </p>
    </figure>
  );
}

interface SensitivityProps {
  feature: Feature | undefined;
  data: Array<{ value: number | string | boolean; prediction: number }> | undefined;
  current: number | string | boolean | undefined;
}

export function SensitivityChart({ feature, data, current }: SensitivityProps) {
  if (!feature) return null;
  if (!data) return <Skeleton className="h-48 w-full" />;
  const numeric = feature.type === "number" || feature.type === "integer";
  const lo = data[0]?.prediction ?? 0;
  const hi = data[data.length - 1]?.prediction ?? 0;
  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="text-sm font-semibold">How {feature.label.toLowerCase()} moves the prediction</figcaption>
      <div className="h-48 w-full" aria-hidden>
        <ResponsiveContainer>
          {numeric ? (
            <LineChart data={data} margin={{ left: 8, right: 16, top: 8, bottom: 16 }}>
              <CartesianGrid stroke="var(--color-data-grid)" />
              <ReferenceArea x1={feature.min ?? undefined} x2={feature.max ?? undefined} fill="var(--color-data-band)" fillOpacity={1} />
              <XAxis dataKey="value" type="number" domain={["dataMin", "dataMax"]} tick={AXIS} tickFormatter={(v: number) => formatValue(v, { ...feature, unit: "" })}
                label={{ value: feature.unit ? `${feature.label} (${feature.unit})` : feature.label, position: "insideBottom", offset: -8, ...AXIS }} />
              <YAxis tick={AXIS} tickFormatter={(v: number) => pct(v, 0)} width={48} />
              <Line dataKey="prediction" stroke="var(--color-data-highlight)" strokeWidth={2} dot={false} isAnimationActive={false} />
              {typeof current === "number" ? <ReferenceLine x={current} stroke="var(--color-border-strong)" strokeDasharray="4 4" /> : null}
            </LineChart>
          ) : (
            <BarChart data={data.map((d) => ({ ...d, value: formatValue(d.value) }))} margin={{ left: 8, right: 16, top: 8, bottom: 4 }}>
              <CartesianGrid vertical={false} stroke="var(--color-data-grid)" />
              <XAxis dataKey="value" tick={AXIS} />
              <YAxis tick={AXIS} tickFormatter={(v: number) => pct(v, 0)} width={48} />
              <Bar dataKey="prediction" fill="var(--color-data-highlight)" isAnimationActive={false} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted-foreground">
        {numeric ? "Shaded band: training range. Dashed line: current value. " : ""}
        Across this range the prediction goes from {pct(lo)} to {pct(hi)}, all else held at the current profile.
      </p>
    </figure>
  );
}
