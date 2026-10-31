import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, ApiError, type ChatTurn, type Feature, type Profile, type SchemaOut } from "@/api/client";
import { streamNarrative } from "@/api/sse";

export const DEBOUNCE_MS = 250;

export function useDebounced<T>(value: T, ms = DEBOUNCE_MS): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return debounced;
}

const retryUnavailable = (count: number, err: unknown) => err instanceof ApiError && err.kind === "unavailable" && count < 2;

export function useSchema() {
  return useQuery({ queryKey: ["schema"], queryFn: ({ signal }) => api.schema(signal), staleTime: Infinity, retry: retryUnavailable });
}

export function useModelInfo(enabled: boolean) {
  return useQuery({ queryKey: ["model"], queryFn: ({ signal }) => api.model(signal), staleTime: Infinity, enabled, retry: retryUnavailable });
}

export function medianProfile(schema: SchemaOut): Profile {
  return Object.fromEntries(schema.features.map((f) => [f.name, f.default as Profile[string]]));
}

/**
 * Debounced what-if call. TanStack Query passes an AbortSignal, so a newer slider position cancels the request
 * in flight, and keepPreviousData leaves the last result on screen with a pending flag instead of blanking it.
 */
export function useWhatIf(scenario: Profile | null, allowExtrapolation: boolean) {
  const debounced = useDebounced(scenario);
  return useQuery({
    queryKey: ["whatif", debounced, allowExtrapolation],
    queryFn: ({ signal }) => api.whatIf({ baseline: null, scenario: debounced as Profile, allow_extrapolation: allowExtrapolation }, signal),
    enabled: debounced != null,
    placeholderData: keepPreviousData,
    retry: retryUnavailable,
  });
}

export function sensitivityGrid(feature: Feature, allowExtrapolation: boolean, points = 25): Array<number | string | boolean> {
  if (feature.type === "boolean") return [false, true];
  if (feature.type === "category") return feature.allowed ?? [];
  const min = feature.min ?? 0;
  const max = feature.max ?? 1;
  const span = max - min;
  const lo = allowExtrapolation ? Math.max(0, min - 0.3 * span) : min;
  const hi = allowExtrapolation ? max + 0.3 * span : max;
  const grid = Array.from({ length: points }, (_, i) => lo + ((hi - lo) * i) / (points - 1));
  return feature.type === "integer" ? [...new Set(grid.map(Math.round))] : grid;
}

export function useSensitivity(feature: Feature | undefined, scenario: Profile | null, allowExtrapolation: boolean) {
  const debounced = useDebounced(scenario);
  return useQuery({
    queryKey: ["sensitivity", feature?.name, debounced, allowExtrapolation],
    queryFn: async ({ signal }) => {
      const grid = sensitivityGrid(feature as Feature, allowExtrapolation);
      const profiles = grid.map((v) => ({ ...(debounced as Profile), [(feature as Feature).name]: v }));
      const out = await api.batch(profiles, true, signal);
      return grid.map((v, i) => ({ value: v, prediction: out.predictions[i].probability ?? out.predictions[i].value ?? 0 }));
    },
    enabled: !!feature && debounced != null,
    placeholderData: keepPreviousData,
    retry: retryUnavailable,
  });
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  source?: "llm" | "template";
}

export type ChatStatus = "idle" | "streaming" | "error";

export function useExplainer() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const [error, setError] = useState<ApiError | null>(null);
  const controller = useRef<AbortController | null>(null);
  const counter = useRef(0);

  const ask = useCallback(
    async (question: string, scenario: Profile, allowExtrapolation: boolean) => {
      controller.current?.abort();
      const ctrl = new AbortController();
      controller.current = ctrl;
      const history: ChatTurn[] = messages.slice(-4).map(({ role, content }) => ({ role, content: content.slice(0, 2000) }));
      const turn = ++counter.current;
      setMessages((m) => [...m, { id: `u${turn}`, role: "user", content: question }, { id: `a${turn}`, role: "assistant", content: "" }]);
      setStatus("streaming");
      setError(null);
      const patchLast = (fn: (msg: ChatMessage) => ChatMessage) => setMessages((m) => [...m.slice(0, -1), fn(m[m.length - 1])]);
      try {
        await streamNarrative(
          { baseline: null, scenario, question, history, allow_extrapolation: allowExtrapolation },
          (e) => {
            if (e.event === "token") patchLast((msg) => ({ ...msg, content: msg.content + e.data.text }));
            if (e.event === "done") patchLast((msg) => ({ ...msg, source: e.data.source }));
            if (e.event === "error") throw new ApiError("server", null, e.data.detail);
          },
          ctrl.signal,
        );
        setStatus("idle");
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setMessages((m) => (m[m.length - 1]?.content ? m : m.slice(0, -1)));
        setError(err instanceof ApiError ? err : new ApiError("server", null, "The explanation could not be produced."));
        setStatus("error");
      }
    },
    [messages],
  );

  useEffect(() => () => controller.current?.abort(), []);
  return useMemo(() => ({ messages, status, error, ask }), [messages, status, error, ask]);
}
