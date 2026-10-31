import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div aria-hidden className={cn("animate-pulse rounded-md bg-muted motion-reduce:animate-none", className)} {...props} />;
}

const badgeVariants = cva("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium [&_svg]:size-3", {
  variants: {
    variant: {
      neutral: "border-border bg-muted text-foreground",
      warning: "border-warning-line bg-warning-soft text-warning",
      danger: "border-danger-line bg-danger-soft text-destructive",
      info: "border-info-line bg-info-soft text-info",
    },
  },
  defaultVariants: { variant: "neutral" },
});

export function Badge({ className, variant, ...props }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

const alertVariants = cva("flex gap-3 rounded-md border p-3 text-sm [&>svg]:mt-0.5 [&>svg]:size-4 [&>svg]:shrink-0", {
  variants: {
    variant: {
      info: "border-info-line bg-info-soft text-foreground [&>svg]:text-info",
      warning: "border-warning-line bg-warning-soft text-foreground [&>svg]:text-warning",
      danger: "border-danger-line bg-danger-soft text-foreground [&>svg]:text-destructive",
    },
  },
  defaultVariants: { variant: "info" },
});

export function Alert({ className, variant, ...props }: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>) {
  return <div role={variant === "danger" ? "alert" : "status"} className={cn(alertVariants({ variant }), className)} {...props} />;
}
