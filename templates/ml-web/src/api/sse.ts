import { send, type ChatTurn, type Profile } from "./client";

export type NarrativeEvent =
  | { event: "meta"; data: { request_id: string; model_version: string; provider: string } }
  | { event: "token"; data: { text: string } }
  | { event: "done"; data: { source: "llm" | "template"; grounded: boolean; reason: string; cached: boolean; unsupported: string[] } }
  | { event: "error"; data: { code: string; detail: string } };

export interface NarrativeBody {
  baseline: Profile | null;
  scenario: Profile;
  question: string;
  history: ChatTurn[];
  allow_extrapolation: boolean;
}

/** POST /v1/explain/narrative and parse its server-sent events as they arrive. */
export async function streamNarrative(body: NarrativeBody, onEvent: (e: NarrativeEvent) => void, signal?: AbortSignal) {
  const res = await send("/v1/explain/narrative", { method: "POST", body: JSON.stringify(body), signal });
  const reader = res.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    let cut = buffer.indexOf("\n\n");
    while (cut >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const fields = Object.fromEntries(
        block.split("\n").map((line) => {
          const at = line.indexOf(": ");
          return [line.slice(0, at), line.slice(at + 2)];
        }),
      );
      if (fields.event && fields.data) onEvent({ event: fields.event, data: JSON.parse(fields.data) } as NarrativeEvent);
      cut = buffer.indexOf("\n\n");
    }
    if (done) return;
  }
}
