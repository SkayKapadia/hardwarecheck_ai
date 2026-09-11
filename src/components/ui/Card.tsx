import { cn } from "@/lib/utils";

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  selected?: boolean;
  disabled?: boolean;
}

export function Card({ selected, disabled, className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "relative p-5 rounded-xl border transition-all duration-300 cursor-pointer overflow-hidden",
        selected
          ? "border-primary/50 bg-primary/5 shadow-md shadow-primary/10 ring-1 ring-primary/20"
          : "border-border/50 bg-card/40 hover:border-border hover:bg-card/60 hover:shadow-sm",
        disabled && "opacity-50 cursor-not-allowed hover:border-border/50 hover:bg-card/40",
        className
      )}
      {...props}
    >
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
}
