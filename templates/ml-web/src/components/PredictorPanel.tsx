import { Info, RotateCcw, Save, TriangleAlert } from "lucide-react";
import { useId } from "react";

import type { Feature, Profile, ProfileValue } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { decimals, formatValue, isOutside } from "@/lib/format";

interface Props {
  features: Feature[];
  profile: Profile;
  allowExtrapolation: boolean;
  onChange: (name: string, value: ProfileValue) => void;
  onReset: () => void;
  onSave: () => void;
  onAllowExtrapolation: (value: boolean) => void;
}

export function PredictorPanel({ features, profile, allowExtrapolation, onChange, onReset, onSave, onAllowExtrapolation }: Props) {
  const switchId = useId();
  return (
    <section aria-labelledby="predictors-heading" className="flex flex-col gap-4">
      <div className="flex flex-col gap-3">
        <h2 id="predictors-heading" className="text-sm font-semibold">
          Predictors
        </h2>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={onReset}>
            <RotateCcw aria-hidden /> Reset to median profile
          </Button>
          <Button variant="outline" size="sm" onClick={onSave}>
            <Save aria-hidden /> Save scenario
          </Button>
        </div>
      </div>
      <div className="flex items-center gap-2 rounded-md bg-muted px-3 py-2">
        <Switch id={switchId} checked={allowExtrapolation} onCheckedChange={onAllowExtrapolation} />
        <Label htmlFor={switchId} className="font-normal text-muted-foreground">
          Allow values outside the training range
        </Label>
      </div>
      <ul className="flex flex-col gap-5">
        {features
          .filter((f) => f.adjustable)
          .map((f) => (
            <li key={f.name}>
              <PredictorControl feature={f} value={profile[f.name]} allowExtrapolation={allowExtrapolation} onChange={(v) => onChange(f.name, v)} />
            </li>
          ))}
      </ul>
    </section>
  );
}

interface ControlProps {
  feature: Feature;
  value: ProfileValue;
  allowExtrapolation: boolean;
  onChange: (value: ProfileValue) => void;
}

export function PredictorControl({ feature, value, allowExtrapolation, onChange }: ControlProps) {
  const id = useId();
  const label = feature.unit ? `${feature.label} (${feature.unit})` : feature.label;
  const header = (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-1">
        <Label htmlFor={id}>{feature.label}</Label>
        {feature.description ? (
          <Tooltip>
            <TooltipTrigger className="rounded-sm text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              <Info aria-hidden className="size-3.5" />
              <span className="sr-only">About {feature.label}</span>
            </TooltipTrigger>
            <TooltipContent>{feature.description}</TooltipContent>
          </Tooltip>
        ) : null}
      </div>
      <output htmlFor={id} className="tabular text-sm font-medium">
        {formatValue(value, feature)}
      </output>
    </div>
  );

  if (feature.type === "number" || feature.type === "integer") {
    const min = feature.min ?? 0;
    const max = feature.max ?? 1;
    const span = max - min;
    const lo = allowExtrapolation ? Math.max(0, min - 0.5 * span) : min;
    const hi = allowExtrapolation ? max + 0.5 * span : max;
    const outside = isOutside(value, feature);
    return (
      <div className="flex flex-col gap-1">
        {header}
        <Slider
          id={id}
          min={lo}
          max={hi}
          step={feature.step ?? 1}
          value={[Number(value)]}
          onValueChange={([v]) => onChange(feature.type === "integer" ? Math.round(v) : Number(v.toFixed(decimals(feature.step))))}
          thumbLabel={label}
          valueText={formatValue(value, feature)}
        />
        <p className="tabular text-xs text-muted-foreground">
          Training range {formatValue(min, { ...feature, unit: "" })} to {formatValue(max, feature)}
        </p>
        {outside ? (
          <p className="flex items-center gap-1 text-xs font-medium text-warning">
            <TriangleAlert aria-hidden className="size-3.5" /> Outside the training range
          </p>
        ) : null}
      </div>
    );
  }

  if (feature.type === "boolean") {
    return (
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor={id}>{feature.label}</Label>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">{value ? "Yes" : "No"}</span>
          <Switch id={id} checked={Boolean(value)} onCheckedChange={onChange} aria-label={feature.label} />
        </div>
      </div>
    );
  }

  const options = feature.allowed ?? [];
  return (
    <div className="flex flex-col gap-1.5">
      {header}
      {options.length <= 4 ? (
        <ToggleGroup id={id} type="single" value={String(value)} onValueChange={(v) => v && onChange(v)} aria-label={feature.label}>
          {options.map((o) => (
            <ToggleGroupItem key={o} value={o}>
              {o.replace(/_/g, " ")}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      ) : (
        <Select value={String(value)} onValueChange={onChange}>
          <SelectTrigger id={id} aria-label={feature.label}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((o) => (
              <SelectItem key={o} value={o}>
                {o.replace(/_/g, " ")}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}
