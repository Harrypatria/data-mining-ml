import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "@/App";
import { ExplainerChat } from "@/components/ExplainerChat";
import { StatusPanel } from "@/components/Panels";
import { ApiError, kindForStatus } from "@/api/client";
import { DEBOUNCE_MS, useExplainer } from "@/hooks/useWorkbench";
import { jsonResponse, model, schema, sseResponse, whatIf } from "./fixtures";

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
}

function routeFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const spy = vi.fn((input: RequestInfo | URL, init?: RequestInit) => Promise.resolve(handler(String(input), init)));
  vi.stubGlobal("fetch", spy);
  return spy;
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("slider to what-if request", () => {
  it("debounces slider moves into one request with the latest value and shows the result", async () => {
    const bodies: Array<Record<string, unknown>> = [];
    const fetchSpy = routeFetch((url, init) => {
      if (url.endsWith("/v1/schema")) return jsonResponse(schema);
      if (url.endsWith("/v1/model")) return jsonResponse(model);
      if (url.endsWith("/v1/whatif")) {
        const body = JSON.parse(String(init?.body));
        bodies.push(body);
        return jsonResponse(whatIf(body.scenario.tenure_months === 40 ? 0.31 : 0.08));
      }
      if (url.endsWith("/v1/predict/batch")) return jsonResponse({ predictions: [] });
      return jsonResponse({}, 404);
    });
    renderApp();
    const thumb = await screen.findByRole("slider", { name: /Tenure/ });
    await waitFor(() => expect(bodies.length).toBe(1));
    expect(thumb).toHaveAttribute("aria-valuetext", "37 months");

    fireEvent.keyDown(thumb, { key: "ArrowRight" });
    fireEvent.keyDown(thumb, { key: "ArrowRight" });
    fireEvent.keyDown(thumb, { key: "ArrowRight" });
    expect(thumb).toHaveAttribute("aria-valuetext", "40 months");

    await waitFor(() => expect(bodies.length).toBe(2), { timeout: DEBOUNCE_MS * 8 });
    expect(bodies[1].scenario).toMatchObject({ tenure_months: 40, plan: "basic", autopay: true });
    expect(bodies[1].allow_extrapolation).toBe(false);
    expect(await screen.findByText("31.0%")).toBeInTheDocument();
    expect(screen.getByText(/Flagged at threshold 22%/)).toBeInTheDocument();
    const whatIfCalls = fetchSpy.mock.calls.filter(([u]) => String(u).endsWith("/v1/whatif"));
    expect(whatIfCalls).toHaveLength(2);
  });

  it("shows the training range and the reset control", async () => {
    routeFetch((url) => (url.endsWith("/v1/schema") ? jsonResponse(schema) : url.endsWith("/v1/model") ? jsonResponse(model) : jsonResponse(whatIf(0.1))));
    renderApp();
    expect(await screen.findByText("Training range 1 to 72 months")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Reset to median profile/ })).toBeInTheDocument();
  });
});

describe("states", () => {
  it("renders the model-not-ready state when the schema request returns 503", async () => {
    routeFetch(() => jsonResponse({ detail: "model not ready" }, 503));
    renderApp();
    expect(await screen.findByRole("heading", { name: "Model not ready" })).toBeInTheDocument();
  });

  it("renders the service-unavailable state when the network fails", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    renderApp();
    expect(await screen.findByRole("heading", { name: "Service unavailable" }, { timeout: 4000 })).toBeInTheDocument();
  });

  it("maps every HTTP status to a designed state", () => {
    expect(kindForStatus(503)).toBe("not_ready");
    expect(kindForStatus(429)).toBe("rate_limited");
    expect(kindForStatus(401)).toBe("unauthorized");
    expect(kindForStatus(422)).toBe("invalid");
    expect(kindForStatus(500)).toBe("server");
    const onRetry = vi.fn();
    render(<StatusPanel error={new ApiError("rate_limited", 429, "Slow down.")} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: /Try again/ }));
    expect(onRetry).toHaveBeenCalled();
  });
});

function ChatHarness() {
  const chat = useExplainer();
  const [auto, setAuto] = useState(false);
  return (
    <ExplainerChat
      messages={chat.messages}
      status={chat.status}
      error={chat.error}
      autoExplain={auto}
      onAutoExplain={setAuto}
      onAsk={(q) => chat.ask(q, { tenure_months: 37, plan: "basic", autopay: true }, false)}
    />
  );
}

describe("streaming explainer chat", () => {
  it("streams tokens into the answer and labels the template fallback", async () => {
    let sent: Record<string, unknown> = {};
    routeFetch((_url, init) => {
      sent = JSON.parse(String(init?.body));
      return sseResponse([
        ["meta", { request_id: "r1", model_version: "1.0.0", provider: "template" }],
        ["token", { text: "The model gives this scenario " }],
        ["token", { text: "a 31.0% probability." }],
        ["done", { source: "template", grounded: false, reason: "provider unavailable", cached: false, unsupported: [] }],
      ]);
    });
    render(<ChatHarness />);
    await userEvent.click(screen.getByRole("button", { name: /Explain this change/ }));
    expect(await screen.findByText("The model gives this scenario a 31.0% probability.")).toBeInTheDocument();
    expect(screen.getByText(/AI explainer unavailable/)).toBeInTheDocument();
    expect(sent).toMatchObject({ question: "Explain this change", baseline: null, allow_extrapolation: false });
    expect(sent).not.toHaveProperty("shap_local");
  });

  it("shows a rate-limit message instead of an answer", async () => {
    routeFetch(() => jsonResponse({ detail: "explanation limit reached" }, 429));
    render(<ChatHarness />);
    const input = screen.getByLabelText("Ask a follow-up question");
    await userEvent.type(input, "Why?");
    await act(async () => {
      await userEvent.click(screen.getByRole("button", { name: "Send question" }));
    });
    expect(await screen.findByText(/explanation limit/)).toBeInTheDocument();
  });
});
