import { Bot, Info, Lightbulb, MessageSquareText, Send } from "lucide-react";
import { useId, useState } from "react";

import type { ChatMessage, ChatStatus } from "@/hooks/useWorkbench";
import type { ApiError } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/feedback";
import { Input, Label } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

const SUGGESTED = [
  "Which predictor matters most for this profile?",
  "What would bring this prediction below the threshold?",
  "How sure is the model about this?",
];

interface Props {
  messages: ChatMessage[];
  status: ChatStatus;
  error: ApiError | null;
  autoExplain: boolean;
  onAutoExplain: (value: boolean) => void;
  onAsk: (question: string) => void;
}

export function ExplainerChat({ messages, status, error, autoExplain, onAutoExplain, onAsk }: Props) {
  const [draft, setDraft] = useState("");
  const inputId = useId();
  const autoId = useId();
  const streaming = status === "streaming";
  const last = messages[messages.length - 1];
  const submit = (q: string) => {
    const question = q.trim();
    if (!question || streaming) return;
    onAsk(question);
    setDraft("");
  };

  return (
    <section aria-labelledby="explainer-heading" className="flex min-h-0 flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 id="explainer-heading" className="flex items-center gap-2 text-sm font-semibold">
          <Bot aria-hidden className="size-4" /> AI explainer
        </h2>
        <div className="flex items-center gap-2">
          <Switch id={autoId} checked={autoExplain} onCheckedChange={onAutoExplain} />
          <Label htmlFor={autoId} className="text-xs font-normal text-muted-foreground">
            Explain after each change
          </Label>
        </div>
      </div>

      <Button onClick={() => submit("Explain this change")} disabled={streaming}>
        <Lightbulb aria-hidden /> Explain this change
      </Button>

      <div className="flex min-h-40 flex-1 flex-col gap-3 overflow-y-auto rounded-md border border-border bg-muted p-3" aria-live="polite" aria-busy={streaming}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-start gap-2 text-sm text-muted-foreground">
            <MessageSquareText aria-hidden className="size-5" />
            <p>Move a predictor, then ask why the prediction changed. Answers use only numbers the service computed.</p>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={m.id} className={m.role === "user" ? "self-end rounded-md bg-primary px-3 py-2 text-sm text-primary-foreground" : "flex flex-col gap-1"}>
              {m.role === "assistant" ? (
                <>
                  <p className="whitespace-pre-line text-sm leading-relaxed">{m.content || (streaming && i === messages.length - 1 ? "Working out the explanation" : "")}</p>
                  {m.source === "template" ? (
                    <p className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Info aria-hidden className="size-3.5" /> AI explainer unavailable; showing the standard explanation built from the same numbers.
                    </p>
                  ) : null}
                </>
              ) : (
                m.content
              )}
            </div>
          ))
        )}
      </div>

      {error ? (
        <Alert variant="danger">
          <Info aria-hidden />
          <p>{error.kind === "rate_limited" ? "You have reached the explanation limit for now. Try again later." : error.message}</p>
        </Alert>
      ) : null}

      <div className="flex flex-wrap gap-2">
        {SUGGESTED.map((q) => (
          <Button key={q} variant="outline" size="sm" className="whitespace-normal text-left" disabled={streaming} onClick={() => submit(q)}>
            {q}
          </Button>
        ))}
      </div>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          submit(draft);
        }}
      >
        <Label htmlFor={inputId} className="sr-only">
          Ask a follow-up question
        </Label>
        <Input id={inputId} value={draft} maxLength={1000} placeholder="Ask a follow-up question" onChange={(e) => setDraft(e.target.value)} />
        <Button type="submit" variant="secondary" size="icon" disabled={streaming || !draft.trim()} aria-label="Send question">
          <Send aria-hidden />
        </Button>
      </form>
      <p className="text-xs text-muted-foreground">
        Explanations describe the model's associations, not causes. {last?.source === "llm" ? "Every number was checked against the service's results." : ""}
      </p>
    </section>
  );
}
