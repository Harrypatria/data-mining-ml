import type { components } from "./schema";

// Types come from the service's OpenAPI document (npm run gen:api), so UI and API cannot drift.
export type Schemas = components["schemas"];
export type Feature = Schemas["FeatureOut"];
export type SchemaOut = Schemas["SchemaOut"];
export type ModelInfo = Schemas["ModelInfo"];
export type WhatIfOut = Schemas["WhatIfOut"];
export type PredictionOut = Schemas["PredictionOut"];
export type Contribution = Schemas["Contribution"];
export type BatchOut = Schemas["BatchOut"];
export type ChatTurn = Schemas["ChatTurn"];
export type ProfileValue = number | string | boolean;
/** Profiles are built from /v1/schema at run time; the server validates every field again. */
export type Profile = Record<string, ProfileValue>;

export const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type ApiErrorKind = "unavailable" | "not_ready" | "rate_limited" | "invalid" | "unauthorized" | "server";

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    public status: number | null,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const MESSAGES: Record<ApiErrorKind, string> = {
  unavailable: "The prediction service cannot be reached. Check that the API is running and try again.",
  not_ready: "The model is still loading or failed verification. Try again shortly.",
  rate_limited: "Too many requests in a short time. Wait a moment, then try again.",
  invalid: "The service rejected these inputs.",
  unauthorized: "The service refused this request. Ask an administrator to check access.",
  server: "The service hit an unexpected error. Try again; if it persists, check the service logs.",
};

export function kindForStatus(status: number): ApiErrorKind {
  if (status === 503) return "not_ready";
  if (status === 429) return "rate_limited";
  if (status === 401 || status === 403) return "unauthorized";
  if (status === 422 || status === 400 || status === 413) return "invalid";
  return "server";
}

let session = "";
export function sessionId(): string {
  if (session) return session;
  try {
    session = sessionStorage.getItem("ml-session") ?? "";
    if (!session) {
      session = crypto.randomUUID();
      sessionStorage.setItem("ml-session", session);
    }
  } catch {
    session = session || `s-${Date.now().toString(36)}`;
  }
  return session;
}

export async function send(path: string, init: RequestInit = {}): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "content-type": "application/json", "x-session-id": sessionId(), ...(init.headers ?? {}) },
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("unavailable", null, MESSAGES.unavailable);
  }
  if (!res.ok) {
    const kind = kindForStatus(res.status);
    const body = await res.json().catch(() => ({}));
    const detail = typeof body?.detail === "string" ? body.detail : Array.isArray(body?.detail) ? body.detail.map((d: { msg?: string }) => d.msg).join("; ") : "";
    throw new ApiError(kind, res.status, kind === "invalid" && detail ? `${MESSAGES.invalid} ${detail}` : MESSAGES[kind]);
  }
  return res;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  return (await send(path, init)).json() as Promise<T>;
}

const post = (body: unknown, signal?: AbortSignal): RequestInit => ({ method: "POST", body: JSON.stringify(body), signal });

export const api = {
  schema: (signal?: AbortSignal) => json<SchemaOut>("/v1/schema", { signal }),
  model: (signal?: AbortSignal) => json<ModelInfo>("/v1/model", { signal }),
  whatIf: (
    body: { baseline: Profile | null; scenario: Profile } & Required<Pick<Schemas["WhatIfRequest"], "allow_extrapolation">>,
    signal?: AbortSignal,
  ) => json<WhatIfOut>("/v1/whatif", post(body, signal)),
  batch: (profiles: Profile[], allowExtrapolation: boolean, signal?: AbortSignal) =>
    json<BatchOut>("/v1/predict/batch", post({ profiles, allow_extrapolation: allowExtrapolation }, signal)),
};
