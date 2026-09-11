import React from "react";
import * as Tooltip from "@radix-ui/react-tooltip";

interface Segment {
  label: string;
  value: number; // in GB
  color: string;
}

interface GaugeProps {
  total: number;
  segments: Segment[];
  size?: number;
}

export function Gauge({ total, segments, size = 200 }: GaugeProps) {
  const center = size / 2;
  const strokeWidth = size * 0.15;
  const radius = center - strokeWidth / 2;
  const circumference = 2 * Math.PI * radius;

  let currentOffset = 0;

  const used = segments.reduce((acc, s) => acc + s.value, 0);
  const percentage = Math.min((used / total) * 100, 100);

  return (
    <Tooltip.Provider>
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Background Ring */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke="var(--muted)"
            strokeWidth={strokeWidth}
          />

          {/* Segments */}
          {segments.map((segment, index) => {
            if (segment.value <= 0) return null;
            
            const segmentPercentage = segment.value / total;
            const strokeDasharray = `${segmentPercentage * circumference} ${circumference}`;
            const strokeDashoffset = -currentOffset * circumference;

            currentOffset += segmentPercentage;

            return (
              <Tooltip.Root key={index}>
                <Tooltip.Trigger asChild>
                  <circle
                    cx={center}
                    cy={center}
                    r={radius}
                    fill="transparent"
                    stroke={segment.color}
                    strokeWidth={strokeWidth}
                    strokeDasharray={strokeDasharray}
                    strokeDashoffset={strokeDashoffset}
                    className="transition-all duration-500 ease-in-out cursor-pointer hover:opacity-80"
                  />
                </Tooltip.Trigger>
                <Tooltip.Portal>
                  <Tooltip.Content
                    className="z-50 px-3 py-1.5 text-sm font-mono border border-border bg-card/95 text-primary shadow-lg"
                    sideOffset={5}
                  >
                    {segment.label}: {segment.value.toFixed(2)} GiB
                    <Tooltip.Arrow className="fill-border" />
                  </Tooltip.Content>
                </Tooltip.Portal>
              </Tooltip.Root>
            );
          })}
        </svg>

        {/* Center Text */}
        <div className="absolute flex flex-col items-center justify-center text-center">
          <span className={`text-2xl font-mono font-bold ${percentage >= 100 ? "text-destructive" : "text-primary"}`}>
            {percentage.toFixed(0)}%
          </span>
          <span className="text-xs font-mono text-muted-foreground mt-1">
            {used.toFixed(1)} / {total.toFixed(0)} GiB
          </span>
        </div>
      </div>
    </Tooltip.Provider>
  );
}
