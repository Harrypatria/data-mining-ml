import type { Feature, Profile, WhatIfOut } from "@/api/client";
import { ContributionChart, SensitivityChart } from "@/components/Charts";
import { Alert } from "@/components/ui/feedback";
import { useSensitivity } from "@/hooks/useWorkbench";

interface Props {
  whatIf: WhatIfOut | undefined;
  features: Feature[];
  moved: Feature | undefined;
  profile: Profile;
  allowExtrapolation: boolean;
}

export function Charts({ whatIf, features, moved, profile, allowExtrapolation }: Props) {
  const sensitivity = useSensitivity(moved, profile, allowExtrapolation);
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <ContributionChart contributions={whatIf?.contributions.scenario} features={features} />
      {sensitivity.error ? (
        <Alert variant="danger">
          <p>{sensitivity.error instanceof Error ? sensitivity.error.message : "The sensitivity curve could not be computed."}</p>
        </Alert>
      ) : (
        <SensitivityChart feature={moved} data={sensitivity.data} current={moved ? profile[moved.name] : undefined} />
      )}
    </div>
  );
}
