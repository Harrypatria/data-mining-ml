import type { Feature, ProfileValue } from "@/api/client";

export function pct(p: number | null | undefined, digits = 1): string {
  return p == null ? "n/a" : `${(p * 100).toFixed(digits)}%`;
}

/** Signed change in percentage points, with an explicit sign so meaning never depends on color. */
export function points(delta: number, digits = 1): string {
  const v = delta * 100;
  const sign = v > 0.05 ? "+" : v < -0.05 ? String.fromCharCode(0x2212) : ""; // typographic minus
  return `${sign}${Math.abs(v).toFixed(digits)} pts`;
}

export function decimals(step: number | null | undefined): number {
  if (!step || step >= 1) return 0;
  return Math.min(4, Math.ceil(-Math.log10(step)));
}

export function formatValue(value: ProfileValue | null | undefined, feature?: Feature): string {
  if (value == null) return "n/a";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    const text = value.toLocaleString(undefined, { maximumFractionDigits: decimals(feature?.step) });
    return feature?.unit ? `${text} ${feature.unit}` : text;
  }
  return value.replace(/_/g, " ");
}

export function isOutside(value: ProfileValue, feature: Feature): boolean {
  return typeof value === "number" && feature.min != null && feature.max != null && (value < feature.min || value > feature.max);
}
