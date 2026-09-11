import { cn } from "@/lib/utils";

interface PillProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
}

export function Pill({ selected, className, children, ...props }: PillProps) {
  return (
    <button
      type="button"
      className={cn(
        "px-4 py-2 border text-sm font-mono transition-all duration-200",
        selected
          ? "border-primary bg-primary/20 text-primary shadow-[0_0_10px_rgba(34,211,238,0.2)]"
          : "border-border bg-card/50 text-muted-foreground hover:text-primary hover:border-primary/50",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
